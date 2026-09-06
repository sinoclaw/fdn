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
                 memory_max=256, plastic_lr=1e-3, plastic_cap=0.1, plastic_decay=0.98,
                 use_plasticity=True, use_memory=True, dynamic_tau=True, dynamic_k=True,
                 warmup_steps=0, warm_temp=3.0, task_constraint=False,
                 n_tasks=4, soft_task_bias=0.0):
        super().__init__()
        self.dim, self.hidden, self.r = dim, hidden, r
        self.base_k, self.kmin, self.kmax, self.ent_scale = base_k, kmin, kmax, ent_scale
        self.plastic_lr, self.plastic_cap, self.plastic_decay = plastic_lr, plastic_cap, plastic_decay
        self.use_plasticity = use_plasticity
        self.use_memory = use_memory
        self.dynamic_tau = dynamic_tau
        self.dynamic_k = dynamic_k
        # Router warm-up + soft→hard curriculum（破「路由鸡生蛋」）
        # warmup_steps>0 时：前 warmup_steps 次训练前向用「全班 soft 路由」（温度从 warm_temp 指数退火到近似 hard），
        # 让每个 Node 都拿到梯度、Router 学到任务亲和；达到 warmup_steps 后切回 hard top-k。
        self.warmup_steps = warmup_steps
        self.warm_temp = warm_temp
        self.step = 0

        self.q = nn.Linear(dim, r)                 # 任务查询向量
        self.nodes = nn.ModuleList()               # 动态 Node 列表（可 spawn 生长）
        self.node_keys = nn.ParameterList()        # 每 Node 独立 key（可生长）
        self.memory = DynamicMemory(r=r, max_entries=memory_max)
        self.register_buffer("h", torch.zeros(0, hidden))           # 每 Node 持续状态
        self.register_buffer("plastic", torch.zeros(0, hidden))     # 每 Node 可塑向量（推理期 ΔW）
        self.register_buffer("usage", torch.zeros(0))               # 每 Node 使用计数
        self.archived = set()                                        # 被 prune 的 Node（不再路由，保留参数）

        # —— FDN-v0.3：Per-Node 生命周期（Protected Expert Formation）——
        # 每个 Node 有 maturity/competence，未成熟者 warm（soft 高门控吸收梯度），成熟者 protected（hard 低可塑）。
        # 节点状态机：NEW -> (warm) WARMING -> LEARNING -> MATURE -> (reactivate) DORMANT/REACTIVATED
        self.register_buffer("maturity", torch.zeros(0))            # (n,) ∈[0,1] 成熟度
        self.register_buffer("competence", torch.zeros(0))          # (n,) 能力/competence 分（usage↑ loss↓ entropy↓ 触发）
        self.register_buffer("node_epoch", torch.zeros(0, dtype=torch.long))  # 每 Node 经历 epoch 数
        # v0.6：Node 级 novelty / error（批次1 纯测量——验证「现有 node 对 B 是否 competence 低」）
        # node_novelty = 1 - cos(q, k)（当前输入离该 node key 多新）；node_error = 该 node 输出的平均预测误差
        self.register_buffer("node_novelty", torch.zeros(0))        # (n,) EMA 新颖度（1-cos，越大越新）
        self.register_buffer("node_error", torch.zeros(0))          # (n,) EMA 预测误差（越大越差）
        self.novelty_ema = 0.5       # node_novelty EMA 更新率
        self.error_ema = 0.5         # node_error EMA 更新率（v0.6：更快反映最近模式，不被长期稀释）
        self.competence_lr = 0.3     # competence 更新率（v0.6：更快适应当前模式，避免 0.05 长期 EMA 稀释）
        self.mature_thr = 0.6         # 成熟阈值：maturity>thr -> MATURE（protected）
        self.mature_epochs = 2        # 至少经过 N 个 epoch 才可能成熟

        # —— FDN-v0.4：任务亲和约束（per-task 专属 Node 组，强制不相交）——
        # 每个 Node 记录归属任务 task_owner（-1=未归属/通用）。当前任务查询只能路由到
        # 「归属当前任务的 Node + 未归属 Node」，归属其他任务的成熟 Node 被屏蔽（protected）。
        # 新任务 spawn 专属 Node 组（task_owner=当前任务）→ A/B/C 各自独占 Node 组，实现真隔离。
        self.register_buffer("task_owner", torch.full((0,), -1, dtype=torch.long))  # (n,) 每 Node 归属任务
        self.current_task = -1        # 当前训练/评估任务 id（由 set_task 注入）
        self.task_constraint = task_constraint   # v0.4 开关：启用强制不相交路由（由 cfg 传入）
        self.n_owned = 0              # 已归属任务的 Node 数（用于隔离检查）

        # —— FDN-v0.4-lite：软任务亲和（soft task bias）——
        # 不硬屏蔽（v0.4 反证已证伪硬隔离），改为给 query 加一个可学习任务偏移 task_emb[task_id]，
        # 软性鼓励 Router 对同分布任务（A/A'）走相似 Node、异分布任务（B/C）偏向不同 Node，但允许复用。
        self.soft_task_bias = soft_task_bias   # 权重 w（0=不启用软偏；>0 启用）
        if soft_task_bias > 0:
            self.task_emb = nn.Embedding(n_tasks, r)
            self.task_emb.weight.data.mul_(0.1)   # 初始小偏移，随训练学习

        for _ in range(initial_nodes):
            self._append_node()

    def _append_node(self, parent_idx=None, noise=0.05):
        n = DynamicNode(self.dim, self.hidden, dynamic_tau=self.dynamic_tau)
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
        # v0.3 生命周期：新 Node 从低成熟度+低 competence 起步（warm 阶段）
        self.maturity = torch.cat([self.maturity, torch.zeros(1)], dim=0)
        self.competence = torch.cat([self.competence, torch.zeros(1)], dim=0)
        self.node_epoch = torch.cat([self.node_epoch, torch.zeros(1, dtype=torch.long)], dim=0)
        # v0.6：新 Node 的 novelty / error 初始为 0（由 forward 选中时用 EMA 更新）
        self.node_novelty = torch.cat([self.node_novelty, torch.zeros(1)], dim=0)
        self.node_error = torch.cat([self.node_error, torch.zeros(1)], dim=0)
        # v0.4：新 Node 归属当前任务（-1=未归属/通用，由 run_one 在任务边界 spawn 时设 current_task）
        owner = torch.tensor([self.current_task], dtype=torch.long)
        self.task_owner = torch.cat([self.task_owner, owner], dim=0)
        if self.current_task >= 0:
            self.n_owned += 1
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
        # v0.4-lite 软任务亲和：给 query 加可学习任务偏移（软性偏向，不屏蔽）
        if self.soft_task_bias > 0 and self.current_task >= 0:
            q = q + self.soft_task_bias * self.task_emb(torch.tensor([self.current_task], device=q.device))
        K = torch.stack([k for k in self.node_keys])              # (n,R)
        scores = torch.einsum("br,nr->bn", q, K) / math.sqrt(self.r)  # (B,n) 任务亲和
        if self.archived:                                          # 屏蔽已归档 Node
            arch_idx = list(self.archived)
            scores[:, arch_idx] = -1e9

        # —— v0.4：任务亲和约束（强制不相交，仅 task_constraint=True 时生效）——
        # 屏蔽归属其他任务的 Node（当前任务 protected 它们），只允许「本任务 Node + 未归属 Node」进入 top-k。
        tm = None
        if self.task_constraint:
            tm = self._task_mask()
            if tm is not None:
                scores = scores.clone()
                scores[:, ~tm] = -1e9
                # 可路由 Node 数受 task mask 限制：k 不得超过「与当前任务亲和可用的 Node 数」
                allow_count = int(tm.sum().item())

        # —— FDN-v0.3：Protected Expert Routing ——
        # 不再做「全班 soft」（v0.2-C 错在让所有 Node 共享梯度）。改做 per-node 保护：
        #   * 未成熟 Node（maturity<thr）-> "warm"：在 top-k 内对其 gate 做 soft 放大（高门控、独立吸收梯度）
        #   * 成熟 Node -> protected：标准 hard gate（top-k softmax），不受后续任务拉宽
        k = self.base_k if not self.dynamic_k else self._dynamic_k(x, scores)  # 但 scores 可能含 -1e9（task mask/archived）
        if tm is not None:
            k = min(k, allow_count)
        k = max(1, min(k, scores.size(-1)))   # 防御：k 至少为 1、不超 Node 数
        top = torch.topk(scores, k, dim=-1)                                # (B,k)
        idx = top.indices                                                  # (B,k)
        gate = torch.softmax(top.values, dim=-1)                           # (B,k)

        # per-node warm 掩码（训练时生效）：未成熟 Node 的门控放大 alpha
        if self.training and hasattr(self, "maturity") and self.maturity.numel() > 0:
            sel_nodes = idx                                            # (B,k)
            mat = self.maturity[sel_nodes.clamp(max=self.maturity.numel()-1)]  # (B,k) 每选中 Node 的成熟度
            warm_mask = (mat < self.mature_thr).float()                # 1=未成熟(warm)
            # 未成熟 Node 门控放大到 alpha（≥1），成熟 Node 保持 1.0
            alpha = 1.0 + (self.warm_temp - 1.0) * warm_mask
            gate = gate * alpha
            gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-8)      # 归一化（保贡献总量）

        # 路由熵（用于 Competence Lock / dynamic k 的合法性检查）
        pp = torch.softmax(scores, dim=-1)
        entropy = float(-(pp * torch.log(pp + 1e-8)).sum(dim=-1).mean().item())

        out_list = []
        mem_ret = self.memory.retrieve(q.detach()) if self.use_memory else torch.zeros(x.shape[0], 1)
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
                # v0.6（批次1 纯测量）：更新被选 Node 的 novelty（1-cos(q,key)）与 error 代理（out 与当前贡献差）
                if self.node_novelty.numel() > ni:
                    qn = q[b:b+1].float()
                    kn = self.node_keys[ni].float()
                    cos_ = torch.dot(qn[0], kn) / ((qn.norm() * kn.norm()) + 1e-8)
                    nov = float(1.0 - cos_)
                    self.node_novelty[ni] = (1 - self.novelty_ema) * self.node_novelty[ni] + self.novelty_ema * nov
                    # error 代理：该 node 输出与 batch 内该样本聚合贡献的偏差幅度（无真 y 下的可测代理）
                    err_proxy = float((out_i.squeeze(0) - contrib).abs()) if contrib is not None else float(out_i.squeeze(0).abs())
                    self.node_error[ni] = (1 - self.error_ema) * self.node_error[ni] + self.error_ema * err_proxy
                if self.use_plasticity:
                    # 可塑权重（推理期 ΔW）：Hebbian 局部更新，带上限+衰减
                    self._plastic_update(ni, h_new.squeeze(0), out_i.squeeze(0),
                                         target=None, scale=float(g.detach()))
                term = g * out_i.squeeze(0)                        # 门控保梯度
                contrib = term if contrib is None else contrib + term
            out_list.append(contrib)
        out = torch.stack(out_list)                                 # (B,1) 含梯度
        if self.use_memory:
            out = out + 0.1 * mem_ret.detach()
        info = {"idx": idx.cpu().numpy(), "k": k,
                "n": len(self.nodes), "warmup": False,
                "mem_keys": self.memory.size,
                "entropy": entropy if self.training else 0.0}
        if self.training:
            self.step += 1
        return out, info

    def set_step(self, step):
        """由训练循环注入当前全局步数（用于 warm 阶段退火阈值判断）。"""
        self.step = int(step)

    def reset_step(self):
        self.step = 0

    # —— FDN-v0.4：任务亲和约束 ——
    def set_task(self, task_id):
        """由 run_one/评估在任务边界注入当前任务 id。路由只允许「归属该任务的 Node + 未归属 Node」。"""
        self.current_task = int(task_id)

    def _task_mask(self):
        """v0.4 隔离 mask：(n,) 逻辑值，True=允许当前任务路由。
        允许：归属 current_task 的 Node + 未归属 Node（task_owner=-1）。
        屏蔽：归属其他任务的 Node（被当前任务 protected）。
        """
        if not self.task_constraint or self.current_task < 0 or self.task_owner.numel() == 0:
            return None   # 未启用约束 → 不屏蔽
        allow = (self.task_owner == self.current_task) | (self.task_owner == -1)
        return allow

    # —— FDN-v0.3：Competence Lock（B）+ Re-activation（C）——
    def update_lifecycle(self, loss, entropy, node_err=None):
        """训练后调用。按 competence 信号更新每 Node maturity（Competence Lock）。
        v0.6：competence 改为「该 node 最近预测误差的倒数」驱动（node_error 越大 → competence 越低，
        即该 node 对当前输入模式外行 = competence gap 高）——这才是 GPT 指的正确信号源。
        之前用 usage_share+全局熵，在未选中 node 上恒 0（competence 死掉），不能反映 competence gap。
        """
        if self.maturity.numel() == 0:
            return
        device = self.maturity.device
        if node_err is None or self.node_error.numel() == 0:
            # 回退：v0.3 逻辑（无 node_error 时）
            usage_share = (self.usage.float() + 1e-8) / (self.usage.float().sum() + 1e-8)
            competence_signal = (1.0 - max(0.0, float(entropy))) * (usage_share * 4.0)
            competence_signal = torch.clamp(torch.as_tensor(competence_signal, device=device), 0.0, 1.0)
        else:
            # v0.6：competence = 归一化 error 的倒数。error 越小 → competence 越高（越能胜任当前模式）。
            # 关键修正：只用「被使用过(usage>0)」的 node 算（未使用的 node error=0 会被误判为金牌，须置中性）。
            err = self.node_error.float()
            used = (self.usage.float() > 0).float()
            # 未使用 node（usage==0）→ 中性 0.5；使用中 node → 归一化 error 倒数
            err_n = err / (err.max() + 1e-8)                # 归一化 [0,1]
            inv = (1.0 - err_n)                              # error 越小 → inv 越大
            competence_signal = torch.where(used > 0, torch.clamp(inv, 0.0, 1.0),
                                            torch.full_like(inv, 0.5))
        self.competence = self.competence * (1 - self.competence_lr) + competence_signal * self.competence_lr
        mature_boost = (self.node_epoch.float() >= self.mature_epochs).float()
        self.maturity = torch.clamp(
            self.maturity + self.competence * 0.15 + mature_boost * 0.05, 0.0, 1.0)
        mature = (self.maturity >= self.mature_thr).float().unsqueeze(-1).expand_as(self.plastic)
        self.plastic = self.plastic * (1.0 - 0.1 * mature)   # 成熟 Node 逐步锁死可塑

    def tick_epoch(self):
        """每个 epoch 结束调用：node_epoch += 1。"""
        self.node_epoch = self.node_epoch + 1

    def is_mature(self, ni):
        """第 ni 个 Node 是否成熟（Competence Lock 判据）。"""
        if ni >= self.maturity.numel():
            return False
        return bool(self.maturity[ni] >= self.mature_thr)

    def reactivate_score(self, q):
        """C. Re-activation：查询 q 对每个成熟 Node 的亲和（用于"找到 A Node 直接恢复"）。
        返回按亲和排序的成熟 Node 索引（Router 据此优先复用已成型模块，而非重新 warm）。
        """
        if self.maturity.numel() == 0:
            return []
        # q 可能传 (16,) 或 (1,16)，统一 reshape 成 (B, r)
        q = q.float().reshape(-1, self.r)
        K = torch.stack([k for k in self.node_keys]).float()          # (n, r)
        Kn = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        cos = torch.einsum("br,nr->bn", qn, Kn).mean(dim=0)           # (n,) 平均亲和
        mature_idx = [i for i in range(self.maturity.numel()) if self.maturity[i] >= self.mature_thr]
        return sorted(mature_idx, key=lambda i: -float(cos[i]))

    def _plastic_update(self, ni, h, out_vec, target, scale):
        """推理期有界 ΔW：plastic += η·scale·(h·sign(err))，再衰减+裁剪。"""
        err = out_vec.detach()                                    # 简化：以输出自身符号近似局部误差
        delta = self.plastic_lr * scale * (torch.sign(err) * h.detach())
        new_p = self.plastic[ni] * self.plastic_decay + delta
        self.plastic[ni] = torch.clamp(new_p, -self.plastic_cap, self.plastic_cap)

    def write_memory(self, x, y):
        if not self.use_memory:
            return
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

    def node_telemetry(self):
        """v0.6：返回每 Node 的 novelty/error/competence/usage/maturity 快照（诊断用）。"""
        d = {}
        if self.node_novelty.numel() > 0:
            d["novelty"] = [round(float(v), 3) for v in self.node_novelty.detach().cpu().tolist()]
            d["error"] = [round(float(v), 3) for v in self.node_error.detach().cpu().tolist()]
        if self.competence.numel() > 0:
            d["competence"] = [round(float(v), 3) for v in self.competence.detach().cpu().tolist()]
        if self.maturity.numel() > 0:
            d["maturity"] = [round(float(v), 3) for v in self.maturity.detach().cpu().tolist()]
        if self.usage.numel() > 0:
            d["usage"] = [float(v) for v in self.usage.detach().cpu().tolist()]
        return d

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
