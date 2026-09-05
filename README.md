# FDN-v0 — Fully Dynamic Network（研究 v0）

> 「训练一个能不断重组、生长、记忆、退化自己的计算系统」的最小实证项目。
> 设计源：一份 ChatGPT 深度设计对话（完整对话存 `docs/DESIGN_SOURCE.md`），设计蓝图与判定准则见 `docs/DESIGN.md`。

## 核心假说

在**路由模块化**网络（每个 Node 是独立计算路径，而非共享投影里的身份）中，模型可以在持续学习里：

> **通过「长出」（spawn）新的 Node 组获取新任务能力，并在任务再次出现时「调用」（re-invoke）原来的 Node 组，而不是重学，从而避免灾难性遗忘。**

这与 GSLM 的关键区别：GSLM 的增长节点只增加拓扑身份、能力落在**共享投影**上（No-Go 根因）。FDN-v0 的 Node 被**逐个路由、各自贡献独立输出头**，拥有独立计算路径——是 GSLM 复盘指出的「真模块化」缺口。

## 六维动态（v0 全部实现）

| 维度 | 实现 | 机制 |
|---|---|---|
| ① 结构动态 | spawn | 新任务分布（q 远离所有 key）→ 从最优父节点「继承+分化」长出新 Node |
| | prune | 使用占比过低 → 归档（不再路由，保留参数） |
| | merge | 两个 key 余弦 > 0.95 → 归档低用量者 |
| ② 参数动态 | 可塑权重 | 推理期对激活 Node 施加有界 Hebbian `ΔW`（衰减+裁剪到 cap） |
| ③ 路由动态 | Dynamic Router | `score = q·key`，每输入 top-k，未选中 Node **不参与计算** |
| ④ 记忆动态 | 外部 KV Memory | write/retrieve/consolidate/forget；检测任务再现 + 辅助输出 |
| ⑤ 时间动态 | 时间常数 | 每 Node `τ_i = sigmoid(U_tau[x;h])`，输入调制 |
| ⑥ 计算动态 | 动态 k | 路由门控熵越高（越难）→ 激活更多 Node |

## 对照模型（控制变量矩阵）

| 模型 | 结构 | 路由 | 增长 | 含义 |
|---|---|---|---|---|
| A | 固定共享 MLP | 无 | 无 | 普通网络：学 A→B→C→A'，A 遗忘 |
| B | 固定 N 节点 MoE | 有 | 无 | 隔离「路由模块化 vs 增长」 |
| D | 固定等容量 MoE | 有 | 无 | 隔离「容量效果」 |
| C | **FDN-v0** | 有 | 有 | 待验证主体 |

## 任务集（toy，驱动路由学任务亲和，不喂 oracle task-id）

`A_add → B_mul → C_logic → A2_add(A')`（核心），可扩展 `D_seq → B_mul`（长序列）。
各任务用「输入分布 + 目标函数」区分。

## 关键指标

准确率（逐任务）、`forgetting_A`、`node_overlap_A_A'`（A 与 A' 激活 Node 的 IoU，核心证据）、
`task_affinity_disjoint_frac`（任务是否由不相交 Node 组承担）、spawn/prune/merge 计数、
`memory_size`、`plastic_magnitude`、`param_count`、`wall_sec`。

## 判定准则

- **Pass**：C 满足 —— (a) A 遗忘 < B 且 < A；(b) `node_overlap_A_A'` 显著 > B；(c) A' 能力 ≈ 初始 A（调用而非重学）。
- **No-Go**：C 未满足以上 → 得到「真模块化 + 生长在当前 toy 体系亦无优势」的有效反证，如实记录交审计。
- 若 B 已优于 A 但 C 无额外增益 → 结论收窄为「路由模块化有用，但生长/记忆/可塑在 toy 规模无额外收益」。

## 诚实声明：新颖点

2026 已有：MoLEM（arXiv:2605.21951）、Dynamic Nested Hierarchies（Frontiers 2026）、
Modular Continual Learning（arXiv:2604.14375）、IBF（arXiv:2604.07108）。
FDN-v0 本体非全新架构，相对新颖点 = 把 6 维动态收进**一个小型化路由模块化网络**并整体验证「长出→调用」机制。

## 运行

```bash
cd /data/fdn && .venv/bin/python experiments/continual.py --seq core --out results/summary_v0.json
# 单测
.venv/bin/python tests/test_fdn.py
```

## 仓库布局

```
model/          DynamicNode / FDN 编排 / Router / DynamicMemory
growth/         EvolutionController（spawn/prune/merge）+ plasticity
experiments/    任务生成器、基线模型、主实验 continually
evaluation/     评估指标（遗忘 / Node 重合 / 亲和 / 参数漂移）
tests/          机制单测
docs/           DESIGN.md（蓝图+判据）+ DESIGN_SOURCE.md（设计源对话）
results/        原始实验 JSON
```

## 结果（跑完更新）

> 见 `results/summary_v0.json`（原始数据）与本 README 下方「实验结论」表。对外审核以原始 JSON + 判定表为准。
