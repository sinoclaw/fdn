"""growth/controller.py — Evolution Controller：结构动态（spawn/prune/merge）。

判据（与 docs/DESIGN.md 对齐）：
- spawn(spawn 阈)：一批样本的 q 到「最佳 Node key」的最大余弦偏低 → 说明来了新任务分布 → 从最优匹配父节点「继承+分化」长出新 Node。
- prune：某 Node 长期使用占比 < prune_usage_阈 → 归档（不再路由，保留参数）。
- merge：两个活跃 Node key 余弦 > merge_cos_阈 → 归档使用占比低者（去重）。

生长时用 optimizer.add_param_group 追加新参数，保留旧优化器状态（GSLM 教训）。记录各动作计数。
"""
import torch


class EvolutionController:
    def __init__(self, eval_every=200, spawn_cos_thr=0.55, prune_usage_thr=0.02,
                 merge_cos_thr=0.95, novelty_warmup=0):
        self.eval_every = eval_every
        self.spawn_cos_thr = spawn_cos_thr
        self.prune_usage_thr = prune_usage_thr
        self.merge_cos_thr = merge_cos_thr
        self.novelty_warmup = novelty_warmup
        self.spawn_count = 0
        self.prune_count = 0
        self.merge_count = 0
        self._steps_since_eval = 0
        self._orig_node_count = None

    def _best_key_cos(self, model, x):
        """一批样本的 q 到最佳 Node key 的余弦（novelty 判据核心）。"""
        q = model.q(x).detach()
        K = torch.stack([k for k in model.node_keys])
        K = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        cos = qn @ K.T                                   # (B,n)
        return cos.max(dim=-1).values.mean().item()      # 均值最大余弦

    def maybe_evolve(self, model, optimizer, x):
        """每个训练 step 后调用；到 eval_every 才评估一次。返回本步有无结构动作。"""
        self._steps_since_eval += 1
        if self._steps_since_eval < self.eval_every:
            return False
        self._steps_since_eval = 0
        if self._orig_node_count is None:
            self._orig_node_count = model.node_count()

        actions = []

        # ① spawn：新任务分布（q 远离所有 key）
        best_cos = self._best_key_cos(model, x)
        if best_cos < self.spawn_cos_thr and model.node_count() - len(model.archived) < 64:
            # 找最优父节点（cos 最高者）
            q = model.q(x).detach()
            K = torch.stack([k for k in model.node_keys])
            K = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
            qn = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
            cos = (qn @ K.T).mean(dim=0)
            parent = int(torch.argmax(cos).item())
            ni = model._append_node(parent_idx=parent, noise=0.05)
            if optimizer is not None:
                optimizer.add_param_group({"params": model.node_params_for_optim(ni)})
            self.spawn_count += 1
            actions.append("spawn")

        # ② merge：两个活跃 key 余弦过高 → 归档低用量者
        active = [i for i in range(model.node_count()) if i not in model.archived]
        if len(active) >= 2:
            for a in range(len(active)):
                for b in range(a + 1, len(active)):
                    i, j = active[a], active[b]
                    c = float(torch.cosine_similarity(model.node_keys[i], model.node_keys[j], dim=0))
                    if c > self.merge_cos_thr:
                        ui = float(model.usage[i]); uj = float(model.usage[j])
                        rm = j if ui >= uj else i
                        model.archived.add(rm)
                        self.merge_count += 1
                        actions.append("merge")

        # ③ prune：使用占比过低 → 归档
        total_u = float(model.usage.sum()) + 1e-8
        active = [i for i in range(model.node_count()) if i not in model.archived]
        for i in active:
            if model.node_count() > 8 and (float(model.usage[i]) / total_u) < self.prune_usage_thr:
                model.archived.add(i)
                self.prune_count += 1
                actions.append("prune")

        return bool(actions)
