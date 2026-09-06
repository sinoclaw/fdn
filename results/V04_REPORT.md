# FDN-v0.4 实验结果报告（V04_REPORT，如实版）

> 本报告与 `results/summary_v04.json`（原始数据）、`model/fdn.py`（任务亲和约束实现）、`experiments/continual.py`（--profile v04）形成三层对账，供外部审计（GPT）核定。
> 依据：v0.3（commit 32066e9）实测 A 遗忘解决（0.845→0）但 B/C 未形成独立 Node 组（reactivation 命中重叠）→ 拍板 v0.4 = 强制任务映射不相交 Node 组。

---

## 0. 一句话结论

**硬隔离（强制不相交）失败——它破坏了 FDN 的核心机制「跨任务复用 Node（Re-activation）」。** v0.4 实测：A 未学会（A_end=0.023）、A 遗忘（forgetting=0.5）、**node_overlap_A_A2=0.0**（A 与 A' 的 Node 完全不相交，违背「A' 调用 A 的 Node」目标）、prune 29 次（硬隔离 spawn 过多无用 Node）。**结论：任务亲和约束应为"软"（鼓励不同任务偏向不同 Node 组但允许复用已演化亲和），而非硬性不相交。**

## 1. 机制（FDN-v0.4 任务亲和约束）

- 每个 Node 记录 `task_owner`（归属任务 id；-1=未归属）。
- `ensure_task_nodes`：任务 0 把初始 Node 划归自己（用足初始容量）；后续任务 `_append_node` spawn `init_per_task` 个专属 Node（动态参数 add_param_group 保留旧 optimizer state）。
- `forward` 的 `_task_mask`：路由只允许「归属当前任务的 Node + 未归属 Node」，归属其他任务的 Node 被屏蔽（protected）。
- `set_task(task_id)` 在任务边界 + 评估逐任务时注入 → 每任务用其专属 Node 组评估。

## 2. 设定

- `--seq core --profile v04 --seed 0 --n_train 1200 --epochs_per_task 40 --init_per_task 8 --run C`
- 对照：v0.3（A 遗忘 0，A_end 0.285，B/C 未独立）。

## 3. 原始数据摘要

| 指标 | v0.3 保护式 | v0.4 硬隔离 |
|---|---|---|
| acc_A_end | 0.285 | **0.023** |
| forgetting_A | 0.0 | **0.5** |
| node_overlap_A_A2 | 0.667 | **0.0** |
| B_mul | 0.051 | 0.031 |
| C_logic | 0.041 | 0.004 |
| final_nodes | 13 | 37（prune 29） |

A_curve=[0.047, 0.0, 0.006, 0.023]

## 4. 判定（重要的方法论级反证）

- **硬隔离（强制不相交）违反 FDN 假说核心**：FDN 的价值正是「同分布任务（A/A'）复用同一批 Node」（v0 已证 IoU=1.0），硬隔离把 A' 强制路由到 task3 专属 Node → `node_overlap_A_A2=0.0` → A 彻底割裂，无法"长出并调用"。
- **每任务固定小容量（init_per_task=8）不足**：A 用初始 12 个 Node 仍学不到 0.285（v0.3 用 13 个共享 Node），说明不是容量绝对不足，而是**隔离切断了同分布任务的容量共享**。
- **prune 29 次**：硬隔离 spawn 大量用不上的专属 Node，浪费+稀释。
- **结论**：v0.3 的「保护旧模块（软）+ 新模块 warm」方向仍是最优；**任务约束应是软偏好（鼓励不同任务偏向不同 Node 组，但允许复用已演化亲和）**，硬性不相交是错误路径。

## 5. 下一步

- **v0.4-lite（软任务亲和）**：保留 v0.3 的 protected + per-node warm，加一个**软 task bias**——为不同任务引入不同 task embedding 偏移（q = W_q·x + task_bias[task]），鼓励 Router 对同分布任务走相似 Node、对异分布任务走不同 Node，**但不硬屏蔽**。既保留跨任务复用，又促进分工。
- 把 v0.4 硬隔离的反证 + v0.3 的正向结果交 GPT 审计，裁定软再偏方案。

## 6. 交付清单

- 代码：`model/fdn.py`（task_owner + set_task + _task_mask + 硬隔离 forward），`experiments/continual.py`（--profile v04 + ensure_task_nodes）。
- 原始数据：`results/summary_v04.json`。
- 单测：`tests/test_fdn.py`（15/15，含 v04 task 约束两测）。
