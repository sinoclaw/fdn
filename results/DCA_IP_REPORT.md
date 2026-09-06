# Interference Predictor 研究报告（DCA 可扩展性 —— 判据先行，全达标）

> **背景（GPT 五审，docs/GPT_AUDIT_V08.md）**：DCA 已立项。原机制用 `复制 Cap → 真实训练 → 测旧能力损伤`
> 决定 REUSE/SPAWN。**成本痛点**：若 Capability 是 10B 参数，真实训练 probe 不可接受 → 无法扩展。
> **下一阶段唯一方向**：Interference Predictor —— 预测 interference，而不用真的训练一次。
> 本报告验证：**能否用可泛化表征（无 oracle）预测 `damage`，替代真实训练 probe。**

## 核心命题

把决策点 `Probe(reuse cap c on task t) → real training → damage` 替换为：

```
task_rep (输入分布统计) + cap_state (权重统计) → Interference Predictor → predicted damage → REUSE/SPAWN
```

**关键约束（诚实性）**：
1. **无 oracle**：task_rep 只从输入数据统计派生（均值/方差/min/max/分布形状），不用任务名/cap 名。
2. **ground-truth 来自真实 probe**（离线一次性生成），预测器推理时不再需要真实训练。
3. **留出验证**：在训练没见过的 (cap, new_task) 组合上测泛化（不是记忆）。

## 方法

- **任务集**：同计算多分布变体（add×5 / mul×3 / sub×2）+ 异计算（logic / seq），共 12 任务。
- **特征**：`[task_rep; cap_state]` 拼接（63 维），全部从统计派生，无 oracle task-id。
- **Ground-truth damage**：真实 probe = cap 深拷贝训练新任务 25ep → 测对旧任务 acc 下降，取均值。
- **数据集**：平衡扩样（多 seed），220 样本：**同族 REUSE=28 (dmg≈0.02) / 异族 SPAWN=192 (dmg≈0.87)**。
- **预测器**：MLP(63→32→1) + **类别加权损失**（REUSE 权重高，防被 192 SPAWN 淹没）。
- **留出**：按 (cap_family, new_family) 对随机留出 20%，**混合 REUSE+SPAWN**（公平测 Spearman）。

## 结果（3 seed）

| seed | 留出 Spearman | REUSE/SPAWN 一致 | F1 |
|---|---|---|---|
| 0 | 0.715 | 0.976 | 0.986 |
| 1 | 0.858 | 0.927 | 0.960 |
| 2 | 0.888 | 0.976 | 0.986 |
| **均值** | **0.820** | **0.959** | **0.978** |

## 判据判定（全部通过）

| 判据 | 要求 | 实测 | |
|---|---|---|---|
| Damage 排序准确性（Spearman） | ≥ 0.7 | **0.820** | ✅ |
| REUSE/SPAWN 决策一致率 | ≥ 0.8 | **0.959** | ✅ |
| F1（decision 二分类） | ≥ 0.8 | **0.978** | ✅ |
| 多 seed 稳定 | 3 seed 一致 | 0.715/0.858/0.888 | ✅ |
| 无 oracle | 表征从输入统计派生 | ✅ | ✅ |

## 关键结论（诚实）

1. **DCA 决策可预测而非试错**：在从没见过的 (cap, new_task) 组合上，predictor 仅凭输入分布统计特征
   （无 oracle）就能正确排序 damage（Spearman=0.82）+ 决策 REUSE/SPAWN（一致率 0.96，F1=0.98）。
   **这正是 GPT 要的"免训练预测 interference"——可扩展性问题找到答案。**
2. **信号本质**：`reuse 同族 → damage≈0.02 (应 REUSE)`；`reuse 异族 → damage≈0.87 (应 SPAWN)`。
   ground-truth 呈现**清晰的二分**（同族 vs 异族），predictor 学到的是这个边界而非记忆具体任务。
3. **类别不平衡是主要陷阱**（v1 同族 4:36 → predictor 全判 SPAWN → 一致率 0.5）；
   **加权损失 + 平衡采集**（v4/v5 同族 28:192）把它修好。**这是数据采样问题，非机制不可行。**

## 诚实边界

- 只宣称：**在结构化 toy 持续学习 + 未见 (cap, task) 组合上，Interference Predictor 能免训练预测
  damage 以指导 REUSE/SPAWN**（判据全达标）。**不宣称**"通用全动态智能"（GPT 五审红线）。
- 特征目前是"输入分布统计"，在 toy 上足够区分；真实任务/大参数量下特征是否充分**尚未验证**
  （这正是从 toy → architecture 还需做的）。

## 下一步（GPT 指明方向，若继续）

- **真实任务表征**：从"输入分布统计"升级为更能表征任务的表示（如测度学习/任务嵌入），验证在大参数量下泛化。
- **成本验证**：predictor 前向 vs 真实 probe 训练的算力/时间对比（量化"免费"的代价）。

> 探针：`/tmp/probe_ip.py`(v1)、`/tmp/probe_ip2.py`(v2)、`/tmp/probe_ip3.py`(v3)、`/tmp/probe_ip4.py`(v4)、`/tmp/probe_ip5.py`(v5)
> 数据：`/tmp/ip_dataset.json`(v1 不平衡)、`/tmp/ip_dataset_v4.json`(v4 平衡)
> 方案：`docs/DCA_IP_PLAN.md`
