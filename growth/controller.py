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
                 max_active=64):
        self.spawn_cos_thr = spawn_cos_thr
        self.prune_usage_thr = prune_usage_thr
        self.merge_cos_thr = merge_cos_thr
        self.young_tasks = young_tasks
        self.freeze_tasks = freeze_tasks
        self.merge_patience = merge_patience
        self.merge_usage_thr = merge_usage_thr
        self.max_active = max_active
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

        # ② spawn：本任务输入分布远离所有 key（novelty）→ 从最优父节点长出新 Node
        best_cos = self._best_key_cos(model, x)
        if best_cos < self.spawn_cos_thr and model.node_count() - len(model.archived) < self.max_active:
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
