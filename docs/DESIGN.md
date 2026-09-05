# FDN-v0 — Fully Dynamic Network（研究 v0 版设计蓝图）

> 由一份 ChatGPT 设计源对话（见 `docs/DESIGN_SOURCE.md`）落地。本文档固定 v0 的假说、架构、对照组、指标与判定准则，供外部审计（GPT）与实际实现对齐。

## 1. 核心假说

在**路由模块化**网络中（每个 Node 是独立计算路径，而非共享投影里的身份），模型可以在持续学习过程中：

> **通过「长出」（spawn）新的 Node 组来获取新任务能力，并在任务再次出现时「调用」（re-invoke）原来的 Node 组，而不是重学，从而避免灾难性遗忘。**

这与 GSLM 的关键区别：GSLM 的增长节点只增加拓扑身份、能力全落在**共享投影**上（No-Go 根因）。FDN-v0 的 Node 被**逐个路由 / 各自贡献输出**，拥有独立计算路径——这是 GSLM 复盘指出的「真模块化」缺口。

## 2. 设计源「六维动态」与 v0 范围

设计源定义 6 维动态。v0 承诺**全部实现但保持在 <100 万参数、CPU 可跑、toy 任务**，先证明机制，不追求能力。

| 维度 | v0 实现 |
|---|---|
| ① 结构动态 | spawn / prune / merge（Evolution Controller 触发） |
| ② 参数动态 | 推理期对 Node 局部可塑参数施加有界 `ΔW`（Hebbian + 衰减 + 上限） |
| ③ 路由动态 | Dynamic Router 按输入选 top-k Node，未选中 Node **不参与计算**（真正动态计算图） |
| ④ 记忆动态 | 外部 Key-Value Memory：write / retrieve / consolidate / forget |
| ⑤ 时间动态 | 每个 Node 有**输入调制**的可学习时间常数 `τ_i = f(x, h)` |
| ⑥ 计算动态 | 简单输入激活少 Node（少算），困难输入激活多 Node（多算）；困难样本允许更多推理步 |

## 3. 架构图（v0 第一闭环）

```
            Input x_t
               │
               ▼
      ┌────────────────┐
      │ Dynamic Router │   → 每输入选 top-k 个 Node（k 随难度可调）
      └───────┬────────┘
              │  Top-K Nodes（仅这些被计算，各自独立输出路径）
   ┌──────────┼──────────┐
   ▼          ▼          ▼
 Node A     Node B     Node C    ← 每个 Node：动态单元 h←h+Δt·f(h,x)，τ_i=f(x,h)
   │          │          │
   └──────────┼──────────┘
              ▼
      Working Memory（write/retrieve/consolidate/forget）
              ▼
          Output head  ← 由「选中 Node 的输出聚合」+「记忆检索」拼装
              ▼
      Evolution Controller（按 loss/novelty/usage/相似度 触发 spawn/prune/merge）
```

## 4. 对照模型（矩阵，镜像 GSLM）

| 模型 | 结构 | 路由 | 增长 | 含义 |
|---|---|---|---|---|
| A | 固定（单 MLP 全共享） | 无 | 无 | 普通网络：学 A→B→C→A'，A 遗忘 |
| B | 固定（同 Node 数，无 spawn/prune） | 有（top-k） | 无 | 隔离「路由模块化」vs「增长」 |
| D | 固定（等容量 = FDN 最终 Node 数） | 有 | 无 | 隔离「容量效果」 |
| C | 动态 | 有 | 有（spawn/prune/merge）+ 记忆 + 可塑权重 | **FDN-v0** |

A/B/D 为确定性基线；C 为待验证主体。

## 5. 任务集（toy，矢量式，可驱动路由学区分）

任务按**输入分布 + 目标函数**区分，Router 从「输入」自身学任务亲和（不喂 oracle task-id）。

| 任务 | 输入分布 | 目标 |
|---|---|---|
| A_add | x=(a,b)∈[0,1]² | a+b（归一化标量） |
| B_mul | x=(a,b)∈[0.5,1]² | a·b |
| C_logic | x∈{0,1}^4 | 多数位/majority |
| D_seq | x=[v0,v1,v2] 等差三元 | 下一项 v3 |
| A'_add 变体 | x=(a,b)∈[0.15,0.85]² | a+b（规则未变，分布略移） |

顺序 **A→B→C→A'**（核心实验），可选加 D→B' 长序列。

## 6. 关键测量指标

- `accuracy`：每个任务（学习时 + 之后 + A' 再现时）的准确率
- `forgetting`：`1 − acc_after / acc_initial`（对 A）
- `active_node_overlap(A, A')`：A 训练期激活 Node 集合 vs A' 再现期激活 Node 集合的 **IoU**（>0 高 = 复用了 A 的 Node，即「长了 A 又调用 A」）
- `node→task affinity`：每个任务是否由**不相交**的 Node 组承担
- `total_nodes / active_nodes / spawn_count / prune_count / merge_count`
- `memory_size`、`flops_per_sample`、`inference_steps`、`parameter_drift`

## 7. 判定准则（先立判据再跑）

- **Pass**：C 满足 —— (a) 对 A 的遗忘 < B 且 < A；(b) 在 A' 上 `active_node_overlap(A,A')` 显著 > B（B 固定结构无法按任务复用，per-task Node 重叠低）；(c) A' 能力 ≈ 初始 A（调用而非重学）。
- **No-Go（GSLM 式反证）**：C 未满足上述任何一项，则「真模块化 + 增长」在当前 toy 体系亦无优势——这是**有效反证**，如实写入报告交审计。
- 若 B 相对 A 已有优势但 C 无增益 → 结论收窄为「路由模块化有用，但生长/记忆/可塑未在 toy 规模带来额外收益」。

## 8. 诚实声明（新颖点）

2026 年已有：MoLEM（arXiv:2605.21951，动态 MoE + 潜记忆）、Dynamic Nested Hierarchies（Frontiers 2026，层增删 + 惊奇调制）、Modular Continual Learning（arXiv:2604.14375，模块化路由 + 冻结专家 + 0 干扰）、IBF（arXiv:2604.07108）。FDN-v0 本体非全新架构，相对新颖点 = 把 6 维动态收进**一个小型化路由模块化网络**并整体验证「长出→调用」机制 —— 这正是 GSLM 缺口所在，值得作为最小实证。
