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

## 结果（2026-09-05，如实版）

**结论：在本次实现的 FDN-v0 + toy 任务集下，核心假说未获支持；机制在「路由层」成立、在「能力层」失败。**（详见 `results/V0_REPORT.md`）

| 模型 | 初始参数 | 最终Node | spawn/prune/merge | A_add 初→末 | A/A' Node重合 | A_add | B_mul | C_logic | A2_add |
|---|---|---|---|---|---|---|---|---|---|
| A (MLP) | 4545 | 1 | — | 1.00→1.00 | 1.0 | **1.00** | 0.09 | 0.385 | 1.00 |
| B (静态12) | 17240 | 12 | — | 0.072→0.125 | 0.70 | 0.125 | 0.062 | 0.078 | 0.096 |
| D (静态24) | 34400 | 24 | — | 0.316→0.574 | 0.667 | 0.574 | 0.014 | 0.096 | 0.570 |
| C (FDN-v0) | 17240 | **54** | **42/5/42** | 0.061→0.055 | **1.0** | 0.055 | 0.0 | 0.010 | 0.092 |

**关键证据**：C 的 A 与 A' 激活 Node 完全重合（IoU=1.0），**Router 学会了"同分布任务复用同一批 Node"** —— '长出并调用'机制在路由层真实成立；但因 spawn/merge 各 42 次、结构震荡过频，承载能力的 Node 从未获得稳定权重 → C 全场最差（A2=0.092，B_mul=0.0）。**失败点与 GSLM 不同**：GSLM 是"节点无独立计算路径"，FDN-v0 是"在线结构演化不够稳定"。

> 原始数据见 `results/summary_v0.json`；判定与局限见 `results/V0_REPORT.md`；实现踩坑见 `results/EXPERIMENT_LOG.md`。对外审核以原始 JSON + V0_REPORT + 代码三层对账。

## 结果（v0.1 / v0.2 演进，2026-09-06 如实版）

| 版本 | 机制 | acc_A_end | forgetting | B | C | 关键结论 |
|---|---|---|---|---|---|---|
| v0 全动态 | 全动态 | 0.055 | — | 0.0 | 0.010 | 结构震荡（spawn/merge 各42） |
| v0.1 稳定化 | 稳定结构+关承重动态 | 0.045 | — | 0.041 | 0.057 | 结构治好，但承重机制被关 → 学不会 |
| v0.2-A 全动态 | 稳定结构+全动态 | 0.211 | 0.0 | 0.002 | 0.008 | 承重动态恢复有效，B/C 仍随机（鸡生蛋） |
| v0.2-B 首任务warm | +soft→hard课程(首任务) | **0.455** | **0.0** | 0.020 | 0.176 | **最有希望**：A 学会+遗忘0+调用 |
| v0.2-C 每任务warm | +每任务全班soft | 0.053 | 0.845 | 0.029 | 0.033 | 灾难性遗忘+结构冻结（全班soft=共享投影覆盖） |

**关键发现（GPT v0.2 复审确认）**：`warm 的 softness` 与 `模块隔离的 sparseness` 矛盾——soft 太多（every warm）→ 覆盖+抑制生长；soft 太少（first/无）→ 冷启动学不动但保留好。**结论**：soft 只能用于"新模块诞生成熟"，旧模块须 protected。→ **v0.3 Protected Expert Formation**。

> 详表见 `results/V02_REPORT.md`、`docs/GPT_AUDIT_V02.md`；原始数据 `summary_v02a/b/c.json`。

## 结果（v0.3 Protected Expert Formation，2026-09-06）

**A 遗忘彻底解决 + B 首次学会，方向正确。** 见 `results/V03_REPORT.md`。
| 指标 | v0.2-B | v0.2-C | v0.3 保护式 |
|---|---|---|---|
| acc_A_end | 0.455 | 0.053 | **0.285** |
| forgetting_A | 0.0 | 0.845 | **0.0** |
| B_mul | 0.020 | 0.029 | **0.051** |
| C_logic | 0.176 | 0.033 | 0.041 |
> 原始数据 `summary_v03.json`；GPT 复审见 `docs/GPT_AUDIT_V02.md`。

## 结果（v0.4 任务亲和约束 Hard-Isolation，2026-09-06）——重要反证

**硬隔离（强制不相交）违背 FDN 核心假说——它切断了同分布任务（A/A'）的 Node 复用。** 见 `results/V04_REPORT.md`。
| 指标 | v0.3 保护式 | v0.4 硬隔离 |
|---|---|---|
| acc_A_end | 0.285 | **0.023** |
| forgetting_A | 0.0 | **0.5** |
| node_overlap_A_A2 | 0.667 | **0.0** |
| final_nodes | 13 | 37（prune 29） |

**结论**：任务亲和约束应为**软偏好**（鼓励不同任务偏向不同 Node 组，允许复用已演化亲和），硬性不相交是错误路径 → v0.4-lite（软 task bias）方向。
> 原始数据 `summary_v04.json`；反证细节见 `results/V04_REPORT.md`、`results/EXPERIMENT_LOG.md`。

## 结果（v0.4-lite 软任务亲和，2026-09-06）—— FDN 迄今最优

**软任务偏存在最优强度 w=0.5——同时做到 A 学会+保留+B 大幅学会+A/A' 复用，是 FDN 迄今最好成绩。** 见 `results/V04LITE_REPORT.md`。
| 版本 | A_end | forget | overlap | B | C | A2 |
|---|---|---|---|---|---|---|
| v0.2-B | 0.455 | 0.0 | 0.636 | 0.020 | 0.176 | 0.447 |
| v0.3 | 0.285 | 0.0 | 0.667 | 0.051 | 0.041 | 0.391 |
| v0.4 硬隔离 | 0.023 | 0.5 | 0.0 | 0.031 | 0.004 | 0.090 |
| **v0.4-lite w=0.5** | **0.545** | **0.0** | 0.700 | **0.209** | 0.002 | **0.477** |

- w=0.3 太弱（A 被 B 挤占 0.104）；w=0.8 太强（task_emb 主导路由破坏 A/A' 复用，A=0.020/遗忘0.75）。
- **机制**：`q = W_q·x + w·task_emb[task]` 软性鼓励不同任务偏向不同 Node 组，但不硬屏蔽（保住 A/A' 复用）。
> 原始数据 `summary_v04lite.json`（w0.3）/`summary_v04lite_w05.json`（w0.5）/`summary_v04lite_w08.json`（w0.8）。

## 结果（v0.5 多 seed + 等难度 + Specialization，2026-09-06）——关键反证

**w=0.5 不跨 seed 稳定；且 FDN 当前架构下任务间无真模块分化（disjoint=0 / 行余弦全>0.88）——模块分化未发生是根本缺口。** 见 `results/V05_REPORT.md`。

| 批次 | 结果 | 判定 |
|---|---|---|
| 1 多 seed (0-5) | A_end 0.018→0.973（50 倍），仅 seed3 达标，B 全 <0.1 | ❌ w=0.5 不复现 |
| 2 等难度 D_sub | D_sub=0.139 >> C_logic=0.002，forgetting=0 | ✅ 换任务有效（GPT 建议证实） |
| 3 Specialization | A~A2=0.984（复用✓）但 A~B=0.927/A~D=0.952（**应低实高**），disjoint=0 | ❌ 无模块分化 |

**核心**：v0.3→v0.4-lite 解决了「A 遗忘」和「w 强度」表层，但「任务如何分化到不同 Node」未解决——这正是 GPT 指向 Dynamic Neural Ecology 的核心（Node 自主 specialization）。
> 原始数据 `summary_v05_seed{0..5}.json` / `summary_v05_equi.json`；报告 `V05_MULTISEED.md`/`V05_REPORT.md`。

## 结果（v0.6 Node 自主专化，2026-09-06）——机制启动但分化未达成

**v0.6 三件机制（Node 级 spawn / Plasticity 5 态 / freeze）成功"启动"，但「模块分化」仍未实现：specialization cos(A,B) 全 >0.99、disjoint 全 0、A 遗忘 6/4 个 >0.8。** 见 `results/V06_REPORT.md`。

| 批次 | 结果 | 判定 |
|---|---|---|
| 1 Node 级状态 | competence 修 bug（error 倒数驱动）+ novelty/error buffer | ✅ 信号源正确 |
| 2 Novelty-Spawn | Node 级综合判据触发 spawn（score 0.512），node12 长出 | ✅ 机制启动，但 node12 被 B/A' 共用 |
| 3 Plasticity 5 态 + freeze | 5 态运作；freeze_old 短跑"救活"A/B（24ep：forget 0.992→0/A 0.449/B 0.904） | ✅ 机制，但 40ep 下 A 学不会是鸡生蛋 |
| 4 多 seed | cos(A,B) 全 >0.99，disjoint 全 0，仅 seed1 偶发 A=0.699 | ❌ 分化未达成 |

**核心**：v0.6 方向不能宣布成功——GPT 的"Node 自主专化"在 FDN 当前架构下**未产生 specialization pressure**（与 v0.5 结论一致，任务间无分化仍是根本缺口）。**深层问题是 A 任务本身没学会（A_curve 起点 0.02，Router 冷启动鸡生蛋）**——下一步应回到"先让 A 学好"（warm-up/curriculum），而非继续加分化机制。
> 原始数据 `summary_v06_seed{0..5}.json`；报告 `V06_REPORT.md`。

## 结果（v0.7 + 最终反证，2026-09-06）——MoE 拓扑架构对 toy 任务梯度分配失败（No-Go 前奏）

**v0.7 competence-gated warm-up 触发两个决定性反证，锁定 FDN 系列最底层根因。** 见 `results/V07_FINAL.md`。

| 版本 | 结果 | 判定 |
|---|---|---|
| v0.7 gated warm-up | A 单任务仍 0.076（soft 路由学不快），触发反证 | ❌ 外围机制不治本 |
| 反证 #1 基线对账 | StaticMLP 单任务 A=**1.000**；StaticMoE=**0.05-0.07** | ❌ MoE 拓扑路由学不会 A |
| 反证 #2 路由可导性 | soft 全连通(可导)仍 0.045-0.09 | ❌ 路由不可导非根因 |

**核心**：**同一个 A 任务，共享 MLP 完美学会（1.0），但任何"多 node 路由 + 独立计算"的 MoE 结构（无论 hard/soft 路由、固定/动态、有无外围机制）都学不会（0.05-0.09）。** 最底层根因是 **MoE 拓扑路由架构对"全局简单/无任务子结构"函数无法梯度分配**——并非路由不可导/结构震荡/动态缺失，而是 MoE 架构本身与 toy 回归任务不适配。**FDN 系列（v0→v0.7）经 8 个 commit、两个反证，收口为"架构层不适配"结论，建议交 GPT 四审裁量终结/转向。**
> 报告 `V07_FINAL.md`；反证探针 `/tmp/probe_baseline.py`、`/tmp/probe_router.py`。
