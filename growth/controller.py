"""growth/controller.py — Evolution Controller v0.1：结构动态的「稳定化补丁」。

承接外部 GPT 审计建议（docs/GPT_AUDIT.md），把 v0 的激进结构演化改为稳定化版本：
- **任务边界触发（降频）**：不再每 batch 评估，改为每个任务结束跑一次 Evolution step。
- **Spawn 冻结期**：新 Node 创建后 `freeze_tasks` 个任务内禁止 prune/merge（否则刚长出来就被干掉）。
- **Merge 长期稳定相似**：cos > 阈 AND 连续 `merge_patience` 任务都满足 AND 双方都非 young AND 被并入方 usage 足够低。
- **Node age / 稳定度**：记录 age、usage_ema、last_spawn_task、last_mut_task；年轻 Node 不参与 prune/merge。
"""
import torch


class EvolutionController:
    def __init__(self, spawn_cos_thr=0.55, prune_usage_thr=0.02, merge_cos_thr=0.95,
                 young_tasks=2, freeze_tasks=2, merge_patience=2, merge_usage_thr=0.05,
                 max_active=64, spawn_patience=2,
                 spawn_comp_thr=0.5, spawn_novel_thr=0.6, spawn_err_thr=0.5,
                 use_node_spawn=False, spawn_score_thr=0.5, spawn_err_thr_ref=5.0):
        self.spawn_cos_thr = spawn_cos_thr
        self.prune_usage_thr = prune_usage_thr
        self.merge_cos_thr = merge_cos_thr
        self.young_tasks = young_tasks
        self.freeze_tasks = freeze_tasks
        self.merge_patience = merge_patience
        self.merge_usage_thr = merge_usage_thr
        self.max_active = max_active
        # v0.6 Novelty-Spawn 参数（Node 级判据）
        self.use_node_spawn = use_node_spawn   # True=Node 级判据（v06 profile）；False=全局 min-cos（v04lite 等向后兼容）
        self.spawn_patience = spawn_patience      # 连续多少任务 competence_gap 持续高才 spawn（防噪声）
        self.spawn_comp_thr = spawn_comp_thr      # competence 低于此阈 = 外行（gap 高）
        self.spawn_novel_thr = spawn_novel_thr    # novelty 高于此阈 = 新模式
        self.spawn_err_thr = spawn_err_thr        # error 高于此阈 = 做得差
        self.spawn_score_thr = spawn_score_thr    # 综合 spawn 分阈值（v0.6 综合判据）
        self.spawn_err_thr_ref = spawn_err_thr_ref  # error 归一化参考（训练正常误差量级）
        self.gap_streak = 0                       # 连续满足 gap 高条件数
        self.last_gap = False                      # 上次是否触发 gap（配合 sustained）
        self.last_spawn_reason = None
        self.task = 0
        self.spawn_count = 0
        self.prune_count = 0
        self.merge_count = 0
        self.node_state = {}       # ni -> dict
        self.pair_stable = {}      # (i,j) -> 连续满足余弦阈值的任务数
        self._usage_prev = None

    def _ensure(self, model):
        for ni in range(model.node_count()):
            if ni not in self.node_state:
                self.node_state[ni] = {"age": 0, "last_spawn_task": -1,
                                       "last_mut_task": -1, "frozen_until": -1}
        # 初始化 usage 基线
        if self._usage_prev is None:
            self._usage_prev = model.usage.clone()

    def _best_key_cos(self, model, x):
        x = torch.from_numpy(x).float() if not torch.is_tensor(x) else x
        q = model.q(x).detach()
        K = torch.stack([k for k in model.node_keys])
        K = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        cos = qn @ K.T
        return cos.max(dim=-1).values.mean().item()

    def _node_competence_gap(self, model, x):
        """v0.6 Node 级 spawn 判据：综合分（非硬 AND）。
        GPT：spawn = f(novelty, error, competence_gap, usage, capacity)。若用硬 AND，A 学会的 node
        competence 高会拉高均值，B 来时永远达不到"普遍低"，spawn 永不触发（本批次实测）。
        改为综合分：score = w_n*avg_novelty + w_e*normalized_avg_error + w_c*(1-avg_competence)。
        返回 (score, avg_novelty, avg_error, score_thr_met)。
        """
        if not hasattr(model, "competence") or model.competence.numel() == 0:
            return None
        comp = model.competence.detach().cpu().float()
        nov = model.node_novelty.detach().cpu().float() if model.node_novelty.numel() > 0 else torch.zeros_like(comp)
        err = model.node_error.detach().cpu().float() if model.node_error.numel() > 0 else torch.zeros_like(comp)
        usage = model.usage.detach().cpu().float()
        used = usage > 0
        if used.sum() < 1:
            self.gap_streak = 0
            return (False, 0.0, 0.0, False, 0.0)
        avg_comp = float(comp[used].mean())
        avg_novel = float(nov[used].mean())
        avg_err = float(err[used].mean())
        # 综合分：novelty 高 + error 高 + competence 低 → 分高，spawn 倾向强
        err_n = min(1.0, avg_err / (self.spawn_err_thr_ref + 1e-8)) if avg_err > 0 else 0.0
        score = 0.4 * min(1.0, avg_novel / self.spawn_novel_thr) + 0.3 * err_n + 0.3 * (1.0 - min(1.0, avg_comp))
        gap = score > self.spawn_score_thr
        if gap:
            self.gap_streak += 1
        else:
            self.gap_streak = 0
        sustained = self.gap_streak >= self.spawn_patience
        self.last_gap = gap
        return (gap, avg_novel, avg_err, sustained, score)

    def _do_spawn(self, model, optimizer, x):
        """从最优父节点（离本任务输入最远的 node）长出一个新的专攻 Node，加入优化器并设冻结期。"""
        q = model.q(x).detach()
        K = torch.stack([k for k in model.node_keys])
        K = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (qn @ K.T).mean(dim=0)
        parent = int(torch.argmax(cos).item())
        ni = model._append_node(parent_idx=parent, noise=0.05)
        if optimizer is not None:
            optimizer.add_param_group({"params": model.node_params_for_optim(ni)})
        self.node_state[ni] = {"age": 0, "last_spawn_task": self.task,
                               "last_mut_task": self.task, "frozen_until": self.task + self.freeze_tasks}
        self.spawn_count += 1
        return ni

    def _task_usage_share(self, model):
        """本任务内各 Node 的 usage 增量占比（相对上次任务边界）。节点增长后补齐 prev。"""
        cur = model.usage.clone().float()
        if self._usage_prev is None:
            self._usage_prev = cur.clone()
        prev = self._usage_prev
        if prev.numel() < cur.numel():
            prev = torch.cat([prev, torch.zeros(cur.numel() - prev.numel(), device=prev.device)])
        delta = cur - prev
        self._usage_prev = cur.clone()
        total = delta.sum().item() + 1e-8
        return delta / total

    def task_boundary(self, model, optimizer, x):
        """每个任务训练结束后调用一次。返回本任务是否发生结构动作。"""
        x = torch.from_numpy(x).float() if not torch.is_tensor(x) else x
        self._ensure(model)
        self.task += 1
        actions = []

        # ① age 递增；降低已冻结阈值（随全局任务计数推进的冻结期）
        for ni, st in self.node_state.items():
            st["age"] += 1

        # ② spawn（v0.6）：Node 级 Novelty-triggered Spawn（use_node_spawn=True 时）。
        # 不再用「全局 min-cos」——那是 Router 视角，A/B 会挤进已有 node 不分化。
        if self.use_node_spawn:
            spawn_dict = self._node_competence_gap(model, x)
            if spawn_dict is not None:
                gap, avg_novel, avg_err, sustained, score = spawn_dict
                capacity_ok = (model.node_count() - len(model.archived)) < self.max_active
                if (gap and sustained and capacity_ok):
                    self._do_spawn(model, optimizer, x)
                    self.last_spawn_reason = {"score": round(float(score), 3), "avg_novelty": round(float(avg_novel), 3),
                                              "avg_error": round(float(avg_err), 3),
                                              "sustained": bool(sustained), "task": self.task}
                    actions.append("spawn")
        else:
            # 回退：全局 min-cos（v04lite 等旧 profile 向后兼容）
            best_cos = self._best_key_cos(model, x)
            if best_cos < self.spawn_cos_thr and model.node_count() - len(model.archived) < self.max_active:
                self._do_spawn(model, optimizer, x)
                self.last_spawn_reason = {"mode": "global_mincos", "best_cos": round(float(best_cos), 3), "task": self.task}
                actions.append("spawn")

        # ③ merge：长期稳定相似（连续 merge_patience 任务 cos>阈 且 双方不 young 且 被并入方 usage 低）
        active = [i for i in range(model.node_count()) if i not in model.archived]
        share = self._task_usage_share(model)
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                i, j = active[a], active[b]
                frozen_i = self.node_state.get(i, {}).get("frozen_until", -1) >= self.task
                frozen_j = self.node_state.get(j, {}).get("frozen_until", -1) >= self.task
                young_i = self.node_state.get(i, {}).get("age", 0) < self.young_tasks
                young_j = self.node_state.get(j, {}).get("age", 0) < self.young_tasks
                if frozen_i or frozen_j or young_i or young_j:
                    continue
                c = float(torch.cosine_similarity(model.node_keys[i], model.node_keys[j], dim=0).detach())
                key = (min(i, j), max(i, j))
                if c > self.merge_cos_thr:
                    self.pair_stable[key] = self.pair_stable.get(key, 0) + 1
                else:
                    self.pair_stable[key] = 0
                    continue
                if self.pair_stable[key] < self.merge_patience:
                    continue
                # 被并入方 = usage 占比更低者
                rm = j if float(share[i]) >= float(share[j]) else i
                if float(share[rm]) < self.merge_usage_thr:
                    model.archived.add(rm)
                    self.node_state[rm]["last_mut_task"] = self.task
                    self.merge_count += 1
                    actions.append("merge")

        # ④ prune：低 usage 且 不 young 且 不冻结
        active = [i for i in range(model.node_count()) if i not in model.archived]
        for i in active:
            st = self.node_state.get(i, {})
            if st.get("frozen_until", -1) >= self.task:
                continue
            if st.get("age", 0) < self.young_tasks:
                continue
            if model.node_count() - len(model.archived) > 6 and float(share[i]) < self.prune_usage_thr:
                model.archived.add(i)
                self.prune_count += 1
                actions.append("prune")

        return bool(actions)
