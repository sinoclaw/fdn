"""model/fdn.py — FDN-v0 编排：Dynamic Router + 动态 Node 列表 + 统一前向 + 记忆接入 + 可塑权重。

关键设计点（对照 GSLM 复盘缺口）：
- 每个 Node 有「独立 key + 独立 in_proj/循环/out_head」，被 Router 选中才参与计算 → 独立计算路径。
- Router 用「查询向量 q=W_q·x」与「每 Node key k_i」的点积做 top-k 路由，key 可随 Node 生长。
- 输出 = Σ 激活 Node 的独立 out 经门控聚合；记忆检索作为辅助项。
- 前向按样本逐条处理（更新每 Node 持续状态 h），保证批次内不同样本可激活不同 Node。
"""
import math
import torch
import torch.nn as nn
from .dynamic_node import DynamicNode
from .memory import DynamicMemory


class FDN(nn.Module):
    def __init__(self, dim=4, hidden=32, r=16, initial_nodes=12,
                 base_k=5, kmin=3, kmax=8, ent_scale=1.0,
                 memory_max=256, plastic_lr=1e-3, plastic_cap=0.1, plastic_decay=0.98):
        super().__init__()
        self.dim, self.hidden, self.r = dim, hidden, r
        self.base_k, self.kmin, self.kmax, self.ent_scale = base_k, kmin, kmax, ent_scale
        self.plastic_lr, self.plastic_cap, self.plastic_decay = plastic_lr, plastic_cap, plastic_decay

        self.q = nn.Linear(dim, r)                 # 任务查询向量
        self.nodes = nn.ModuleList()               # 动态 Node 列表（可 spawn 生长）
        self.node_keys = nn.ParameterList()        # 每 Node 独立 key（可生长）
        self.memory = DynamicMemory(r=r, max_entries=memory_max)
        self.register_buffer("h", torch.zeros(0, hidden))           # 每 Node 持续状态
        self.register_buffer("plastic", torch.zeros(0, hidden))     # 每 Node 可塑向量（推理期 ΔW）
        self.register_buffer("usage", torch.zeros(0))               # 每 Node 使用计数
        self.archived = set()                                        # 被 prune 的 Node（不再路由，保留参数）

        for _ in range(initial_nodes):
            self._append_node()

    def _append_node(self, parent_idx=None, noise=0.05):
        n = DynamicNode(self.dim, self.hidden)
        if parent_idx is not None:
            # 从父节点继承一部分 + 噪声分化（继承+分化）
            with torch.no_grad():
                for pm in ("in_proj", "cell_h", "cell_x", "tau", "out_head"):
                    src = getattr(self.nodes[parent_idx], pm)
                    dst = getattr(n, pm)
                    dst.weight.copy_(src.weight + noise * torch.randn_like(src.weight))
                    if dst.bias is not None:
                        dst.bias.copy_(src.bias)
        self.nodes.append(n)
        key = torch.randn(self.r)
        if parent_idx is not None:
            key = self.node_keys[parent_idx].detach() + noise * torch.randn(self.r)
        self.node_keys.append(nn.Parameter(key))
        # 状态/可塑/用量随 Node 增长
        self.h = torch.cat([self.h, torch.zeros(1, self.hidden)], dim=0)
        self.plastic = torch.cat([self.plastic, torch.zeros(1, self.hidden)], dim=0)
        self.usage = torch.cat([self.usage, torch.zeros(1)], dim=0)
        return len(self.nodes) - 1

    def _grow_keys(self):
        pass  # node_keys 已是 ParameterList，天然支持生长

    def _dynamic_k(self, x, scores):
        """计算动态 k：路由门控熵越高（越难分）→ 激活越多 Node（计算动态⑥）。"""
        p = torch.softmax(scores, dim=-1)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)          # (B,)
        mean_ent = entropy.mean().item()
        k = int(round(self.base_k + self.ent_scale * mean_ent))
        return max(self.kmin, min(self.kmax, k))

    def _mark_used(self, sel):
        for ni in sel:
            self.usage[ni] = self.usage[ni] + 1

    def forward(self, x, reset=False):
        """x:(B,DIM)。返回 out:(B,1) 与 info(激活 Node 索引/k/难度/记忆检索量)。"""
        B = x.shape[0]
        if reset:
            self.h.zero_()
        q = self.q(x)                                             # (B,R)
        K = torch.stack([k for k in self.node_keys])              # (n,R)
        scores = torch.einsum("br,nr->bn", q, K) / math.sqrt(self.r)  # (B,n) 任务亲和
        if self.archived:                                          # 屏蔽已归档 Node
            arch_idx = list(self.archived)
            scores[:, arch_idx] = -1e9
        k = self._dynamic_k(x, scores)

        top = torch.topk(scores, k, dim=-1)                       # (B,k)
        idx = top.indices                                          # (B,k)
        gate = torch.softmax(top.values, dim=-1)                   # (B,k)

        out_list = []
        mem_ret = self.memory.retrieve(q.detach())                 # (B,1) 记忆辅助（不进梯度）
        activations = []
        for b in range(B):
            sel = idx[b].tolist()
            activations.append(sel)
            self._mark_used(sel)
            contrib = None
            for j, ni in enumerate(sel):
                g = gate[b, j]                                     # 张量（保 Router 梯度）
                h_old = self.h[ni].detach().clone()                 # 复制成独立张量，避免被后续 inplace 写入污染梯度
                h_new, out_i = self.nodes[ni](h_old.unsqueeze(0), x[b:b+1])
                self.h[ni] = h_new.squeeze(0).detach()
                # 可塑权重（推理期 ΔW）：Hebbian 局部更新，带上限+衰减
                self._plastic_update(ni, h_new.squeeze(0), out_i.squeeze(0),
                                     target=None, scale=float(g.detach()))
                term = g * out_i.squeeze(0)                        # 门控保梯度
                contrib = term if contrib is None else contrib + term
            out_list.append(contrib)
        out = torch.stack(out_list)                                 # (B,1) 含梯度
        out = out + 0.1 * mem_ret.detach()
        info = {"idx": idx.cpu().numpy(), "k": k, "n": len(self.nodes),
                "mem_keys": self.memory.size}
        return out, info

    def _plastic_update(self, ni, h, out_vec, target, scale):
        """推理期有界 ΔW：plastic += η·scale·(h·sign(err))，再衰减+裁剪。"""
        err = out_vec.detach()                                    # 简化：以输出自身符号近似局部误差
        delta = self.plastic_lr * scale * (torch.sign(err) * h.detach())
        new_p = self.plastic[ni] * self.plastic_decay + delta
        self.plastic[ni] = torch.clamp(new_p, -self.plastic_cap, self.plastic_cap)

    def write_memory(self, x, y):
        q = self.q(x).detach()
        self.memory.write(q, torch.as_tensor(y, dtype=torch.float32, device=x.device))

    def active_node_set(self):
        """最近一次前向被激活的 Node 集合（由调用方用 info['idx'] 记录）。"""
        return set()

    def node_count(self):
        return len(self.nodes)

    def node_params_for_optim(self, ni):
        """返回编号为 ni 的 Node 的参数字典（供 add_param_group 用，保留旧优化器状态）。"""
        params = list(self.nodes[ni].parameters()) + [self.node_keys[ni]]
        return params

    def key_cosine(self):
        """所有活跃 Node key 两两余弦（用于 merge 判据）。"""
        K = torch.stack([k for k in self.node_keys])
        K = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
        return (K @ K.T)

    def param_count(self):
        return sum(int(p.numel()) for p in self.parameters())

    def params_dict(self):
        d = {}
        for i, nd in enumerate(self.nodes):
            for k, v in nd.state_dict().items():
                d[f"node{i}.{k}"] = v
        return {k: v.detach().cpu().numpy() for k, v in d.items()}
