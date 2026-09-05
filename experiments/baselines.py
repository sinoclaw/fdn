"""experiments/baselines.py — 确定性对照模型（A/B/D），与 FDN-v0（C）做控制变量对比。

- A（static MLP）：单层共享网络，无路由、无模块隔离。代表「普通网络学 A→B→C→A'，A 遗忘」。
- B（static MoE，固定 N 节点）：有路由模块化，但结构固定（无 spawn/prune/merge），无记忆、无可塑权重。
     等价于去掉增长/记忆/可塑后的 FDN → 隔离「路由模块化 vs 增长」。
- D（static MoE 等容量）：结构与 B 相同，但 Node 数 = FDN 最终 Node 数 → 隔离「容量效果」。

三者共享同一 DynamicNode 结构，保证除「是否动态」外可比。
"""
import math
import torch
import torch.nn as nn
from model.dynamic_node import DynamicNode


class StaticMLP(nn.Module):
    """A：普通共享 MLP，回归到 [0,1]。"""
    def __init__(self, dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, reset=False):
        return self.net(x)

    def param_count(self):
        return sum(int(p.numel()) for p in self.parameters())

    def params_dict(self):
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}


class StaticMoE(nn.Module):
    """B/D：固定结构路由模块化网络（无增长/记忆/可塑）。"""
    def __init__(self, dim=4, hidden=32, r=16, n_nodes=12, k=5):
        super().__init__()
        self.dim, self.hidden, self.r = dim, hidden, r
        self.k = k
        self.q = nn.Linear(dim, r)
        self.nodes = nn.ModuleList([DynamicNode(dim, hidden) for _ in range(n_nodes)])
        self.node_keys = nn.ParameterList([nn.Parameter(torch.randn(r)) for _ in range(n_nodes)])
        self.register_buffer("h", torch.zeros(n_nodes, hidden))
        self.register_buffer("usage", torch.zeros(n_nodes))

    def forward(self, x, reset=False):
        if reset:
            self.h.zero_()
        q = self.q(x)                                              # (B,R)
        K = torch.stack([k for k in self.node_keys])               # (n,R)
        scores = torch.einsum("br,nr->bn", q, K) / math.sqrt(self.r)
        top = torch.topk(scores, self.k, dim=-1)
        idx, gate = top.indices, torch.softmax(top.values, dim=-1)
        B = x.shape[0]
        out_list = []
        for b in range(B):
            sel = idx[b].tolist()
            self.usage[sel] += 1
            contrib = None
            for j, ni in enumerate(sel):
                h_new, out_i = self.nodes[ni](self.h[ni].detach().clone().unsqueeze(0), x[b:b+1])
                self.h[ni] = h_new.squeeze(0).detach()
                term = gate[b, j] * out_i.squeeze(0)               # 门控保梯度（与 FDN 一致，公平对比）
                contrib = term if contrib is None else contrib + term
            out_list.append(contrib)
        out = torch.stack(out_list)                                # (B,1)
        return out, {"idx": idx.cpu().numpy(), "k": self.k, "n": len(self.nodes)}

    def param_count(self):
        return sum(int(p.numel()) for p in self.parameters())

    def params_dict(self):
        d = {}
        for i, nd in enumerate(self.nodes):
            for k, v in nd.state_dict().items():
                d[f"node{i}.{k}"] = v
        return {k: v.detach().cpu().numpy() for k, v in d.items()}
