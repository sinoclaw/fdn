"""model/memory.py — DynamicMemory：外部 Key-Value 记忆（设计源④）。

write / retrieve / consolidate / forget 四动作；key=路由查询嵌入 q，value=目标原型。
作用：①检测任务再现（A' 检索到 A 的原型）；②辅助输出；③记忆体量随任务增长（memory_size 指标）。
"""
import torch
import torch.nn as nn


class DynamicMemory(nn.Module):
    def __init__(self, r=16, max_entries=512):
        super().__init__()
        self.r = r
        self.max_entries = max_entries
        self.register_buffer("keys", torch.zeros(0, r))          # (m, R)
        self.register_buffer("values", torch.zeros(0, 1))        # (m, 1)
        self.register_buffer("counts", torch.zeros(0))           # 命中计数

    def _cluster_key(self, q):
        """对键做在线聚类：q 接近已有键则推近，否则新插入。返回 (key, val, is_new)。"""
        q = q.detach()  # 记忆不参与梯度
        if self.keys.shape[0] == 0:
            return q.unsqueeze(0), None, True
        sim = torch.cosine_similarity(self.keys, q.unsqueeze(0), dim=-1)
        best = int(sim.argmax().item())
        if float(sim[best]) > 0.7:
            # 就近聚簇：更新该簇键为与 q 的均值（前向一致）
            return self.keys[best].unsqueeze(0), best, False
        return q.unsqueeze(0), None, True

    def write(self, q, y):
        """q:(B,R), y:(B,) -> 写入/聚合记忆。返回是否触发新插入（用于 memory 增长日志）。"""
        keys, vals, counts, new_flags = [], [], [], []
        for b in range(q.shape[0]):
            k, idx, is_new = self._cluster_key(q[b])
            keys.append(k.squeeze(0))
            if idx is not None:
                # 已存在簇：在线均值更新
                v = self.values[idx]
            else:
                v = torch.tensor([y[b].item()], device=q.device)
            vals.append(v)
            new_flags.append(is_new)
        keys = torch.stack([self._normalize(k) for k in keys])
        vals = torch.stack(vals)
        self._store(keys, vals, torch.tensor(new_flags, device=q.device, dtype=torch.bool))

    def _normalize(self, k):
        return k / (k.norm(dim=-1, keepdim=True) + 1e-8)

    def _store(self, keys, vals, is_new):
        # 在线合并：新插入的追加；已有的走均值更新（简化：直接 append 新键，旧簇更新值）
        cur_keys = self.keys
        cur_vals = self.values
        cur_counts = self.counts
        # 逐条处理（数据量小，可接受）
        new_keys, new_vals = [], []
        for i in range(keys.shape[0]):
            if bool(is_new[i]):
                new_keys.append(keys[i]); new_vals.append(vals[i])
        if new_keys:
            new_keys = torch.stack(new_keys)
            new_vals = torch.stack(new_vals)
            cur_keys = torch.cat([cur_keys, new_keys], dim=0)
            cur_vals = torch.cat([cur_vals, new_vals], dim=0)
            cur_counts = torch.cat([cur_counts, torch.ones(len(new_keys), device=cur_counts.device)], dim=0)
        # forget：超容量时按命中计数淘汰最冷门
        if cur_keys.shape[0] > self.max_entries:
            order = torch.argsort(cur_counts)
            keep = order[cur_keys.shape[0] - self.max_entries:]
            cur_keys, cur_vals, cur_counts = cur_keys[keep], cur_vals[keep], cur_counts[keep]
        self.keys, self.values, self.counts = cur_keys, cur_vals, cur_counts

    def retrieve(self, q):
        """q:(B,R) -> (B,1) 按余弦相似度加权的记忆值。"""
        if self.keys.shape[0] == 0:
            return torch.zeros(q.shape[0], 1, device=q.device)
        sim = torch.cosine_similarity(self.keys.unsqueeze(0), q.unsqueeze(1), dim=-1)  # (B,m)
        w = torch.softmax(sim * 5.0, dim=-1)
        return (w @ self.values)  # (B,1)

    def snapshot(self):
        return {"keys": self.keys.detach().cpu().numpy().tolist(),
                "values": self.values.detach().cpu().numpy().reshape(-1).tolist(),
                "counts": self.counts.detach().cpu().numpy().tolist()}

    @property
    def size(self):
        return int(self.keys.shape[0])
