# FDN-v0 实验日志（EXPERIMENT_LOG）

> 记录 FDN-v0 从设计到实现再到实验的全过程。给外部审计（GPT）对齐「设计 → 实现 → 结果」三层。

## 2026-09-05 项目启动

### 源起
- 用户给出一份 ChatGPT 深度设计对话（存 `docs/DESIGN_SOURCE.md`），提出 **FDN-v0（Fully Dynamic Network）**：一个能自主改变结构/参数/路由/记忆/时间尺度/计算量的全动态网络。
- 明确要求：与 GSLM 项目同款流程——建公开 GitHub 仓库、写码、跑实验、推结果、交外部 GPT 审。

### 文献查重（先查重定位新颖点）
2026 年已有高度重叠工作：
| 来源 | 覆盖的 FDN 机制 |
|---|---|
| MoLEM（arXiv:2605.21951） | 动态 MoE + 潜记忆注入 + 冻结基座 |
| Dynamic Nested Hierarchies（Frontiers 2026） | 层级增删 + 惊奇信号调频（终身学习自进化） |
| Modular Continual Learning（arXiv:2604.14375） | 模块化路由 + 冻结专家 + 0 干扰 |
| IBF（arXiv:2604.07108） | 连续学习动力学理论 |

**结论**：FDN-v0 本体不是全新架构。相对新颖点 = 把 6 维动态收进**一个小型化路由模块化网络**，并整体验证「长出能力组 → 再调用而非重学」这一机制。该点正是 GSLM No-Go 复盘指出的**真模块化缺口**（GSLM 增长节点无独立计算路径、能力落在共享投影）。故 FDN-v0 值得做最小实证（不投 GPU，CPU 可复现）。

### 设计蓝图
见 `docs/DESIGN.md`（假说、架构图、A/B/C/D 对照矩阵、任务集、指标、判定准则）。

### 实现要点（对照 GSLM 缺口的关键修复）
- **每个 Node = 完整独立子网**（in_proj + 循环单元 + 独立 out_head + 独立 router key），被 Router 选中才参与计算 → 真正独立计算路径，而非共享投影里的身份。
- **Router**：查询向量 `q=W_q·x` 与每 Node key `k_i` 点积做 top-k 路由；gate 是软 softmax，**门控保留梯度**（否则 Router 学不到任务亲和）。
- **Evolution Controller**：spawn（新任务分布 → 继承+分化长出新 Node）/ prune（低用量归档）/ merge（key 高余弦去重）；生长用 `optimizer.add_param_group` 保留旧优化器状态。
- **记忆**：外部 KV Memory，训练时 write，供任务再现检测 + 辅助输出。
- **可塑权重**：推理期有界 Hebbian `ΔW`（衰减+裁剪）。
- **时间**：每 Node `τ_i=sigmoid(U_tau[x;h])`。
- **计算**：路由门控熵越高激活越多 Node（k 动态）。

### 实现过程踩坑（已修复）
1. `self.h` buffer 的**视图复用**导致 autograd inplace 版本错 → 读状态时 `detach().clone()`。
2. `torch.stack(out_list).unsqueeze(1)` 多包一层 → 与 mem_ret 相加触发广播维度错 → 去掉 `unsqueeze`。
3. 基线 `float(gate)` 丢掉 Router 梯度，且 `outs[b]=contrib` 丢梯度 → 改为与 FDN 相同的「含梯度张量 + stack」，保证 A/B/D/C 对比公平。
4. 单任务诊断暴露**状态未清零**导致回归污染 → toy 独立回归任务改为每次前向 `reset=True`；并补上训练时 `write_memory`。

### 单元测试
`tests/test_fdn.py`：9/9 通过（结构 spawn 真增长、prune/merge 改 active 集合、plastic 有界、路由随输入不同、高熵输入 k 更大、τ 不同、指标正确）。

### 运行
- 环境：`/data/fdn/.venv`（torch 2.14 CPU，清华源）
- 命令：`.venv/bin/python experiments/continual.py --seq core --seed 0 --n_train 1200 --epochs_per_task 40 --run all --out results/summary_v0.json`
- 顺序任务 A_add→B_mul→C_logic→A2_add，40 epoch/任务，对比 A/B/D/C。

（实验完成后在本文件尾部追加结果与判定。）

## 2026-09-05 实验完成（seed=0，40ep/任务）

命令：`.venv/bin/python experiments/continual.py --seq core --seed 0 --n_train 1200 --epochs_per_task 40 --run all --out results/summary_v0.json`

### 结果（摘要，完整见 V0_REPORT.md 与 summary_v0.json）
| 模型 | 最终Node | spawn/prune/merge | A_add 末 | A/A'重合 | 判定 |
|---|---|---|---|---|---|
| A (MLP) | 1 | — | 1.00 | 1.0 | 学得最好，动态缺失 |
| B (静态12) | 12 | — | 0.125 | 0.70 | 静态 MoE 中游 |
| D (静态24) | 24 | — | 0.574 | 0.67 | 容量大 → 略好 |
| C (FDN-v0) | 54 | 42/5/42 | 0.055 | **1.0** | **全场最差** |

### 判定：假说未获支持；失败点与 GSLM 不同
- **成立**：全动态机制真实发生（结构 12→54、spawn/merge 各42、记忆 51 条、可塑 0.049）；Router 学会"同分布任务复用同批 Node"（A/A' Node 集合完全一致，IoU=1.0）。
- **失败**：结构震荡（每 120 batch 评估 + spawn_cos_thr=0.55 + merge_cos_thr=0.95）压过学习节奏 → 承载能力的目标 Node 无稳定权重 → C 全场最差。
- **对比**：GSLM 根因=节点无独立计算路径；FDN-v0 根因=在线结构演化不稳。**不是同一失败**，本轮修复了 GSLM 缺口却暴露新缺口。

### 局限
序列以加法变体收尾污染"forgetting on A"判据（实际对比应为"终点 A_add"）；MoE 族本身比 MLP 难学（B/D/C 均远低于 A）；仅单 seed；toy 容量充足未做容量成瓶颈档位。

### 待办（交 GPT 裁量）
若给假说更公平机会 → 改实现（任务边界触发结构评估、降 merge 率、加节点冻结/毕业），另立版，非本版范围。

## 2026-09-05 FDN-v0.1 Stability Patch（按 GPT 审计）——已完成并跑通

按 docs/GPT_AUDIT.md 落地 v0.1：
- 关 Hebbian plasticity、关 memory、固定 τ、固定 k（只留 Dynamic Router + Dynamic Structure + Static Node Weights）
- Structure 稳定化：任务边界触发 Evolution、Spawn 冻结期（freeze_tasks）、Merge 长期稳定（merge_patience + 非 young + 低 usage）、Node age/stability 记录

命令：`.venv/bin/python experiments/continual.py --profile v01 --seed 0 --n_train 1200 --epochs_per_task 40 --run C --out results/summary_v01.json`

### v0.1 vs v0（均 40ep 顺序 A→B→C→A'）
| | 最终Node | spawn/prune/merge | mem | acc_A末 | overlap_A_A2 | 说明 |
|---|---|---|---|---|---|---|
| v0 全动态 | 54 | 42/5/42 | 51 | 0.055 | 1.0 | 结构震荡 |
| v0.1 稳定化 | 15 | 3/6/**0** | 0 | 0.045 | 0.667 | merge 震荡被治住，但学习未救回 |

### v0.1 判定（如实）
- **结构稳定化目标达成**：merge 0、节点 15、spawn 3 —— GPT 要的"停止 spawn→merge 震荡、冻结年轻节点"实现并验证。
- **但学习未提升**：v0.1 最终 acc 各任务 ~0.04-0.06（与 v0 同级、近随机）；`acc_A_init=0.016`，即**"Router+结构+静态权重"最小组合连任务 A（40ep）都学不会**。
- **单任务对照**（全动态开关、30ep、无 controller，1200 样本）：A_add 可达 **0.598** —— 证明这些动态（memory/plasticity/τ/动态k）对"能学会"是承重的，GPT 建议的隔离最小组合把承重机制也抽掉了。
- **定性**：v0.1 部分**证伪**了 GPT 的假设（结构不稳定非唯一根因）——即使结构完全稳定 + 最小组合，路由模块化 FDN 在该 toy 任务上仍无法学到可用精度。失败更可能是"路由模块化 + 逐样本 top-k 门控"的优化本身脆弱，而非仅结构震荡。

### 局限
单 seed；toy 任务容量充足、MoE 族本就比稠密 MLP 难学（A=1.0 vs D=0.574）；消融在短 epoch 下噪声大（15ep 连全动态都只 0.074），未做大样本长 epoch 消融定量。

### 下一步待议（交 GPT）
"能学会"的承重机制（memory/plasticity/τ/k）与该机制"学会怎么做"的区分，是 v0.1 给出的新问题：最小组合学不动，说明隔离法需改为"保留承重动态、只关非承重的结构/路由变量"，或以更难/更稳任务重测。


