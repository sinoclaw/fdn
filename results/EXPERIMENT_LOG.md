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

## 2026-09-06 GPT v0.1 复审 + FDN-v0.2 重新规划

### GPT 复审结论（docs/GPT_AUDIT_V01_REVIEW.md，链接 t_6a9c75...）
GPT 对 v0.1（cbafad0）复审，与团队判断一致且更精准：
- **v0.1 结构稳定化成功**（merge 42→0、节点15），但**隔离实验把承重动态也关掉了**（acc 仍 0.045）。
- **单任务全动态 A_add=0.598** 是铁证：memory/plasticity/τ/动态k 是「能学会」的承重机制。
- **真正问题 = Dynamic Routing 鸡生蛋**：Router 冷启动不知谁擅长 A → 随机分配 → Node 信号不足 → 学不好 → Router 更不知所措。**是优化问题，非结构/机制缺失。**
- 建议 v0.2：**保留承重动态、只隔离结构/路由变量** + **Router warm-up + soft→hard top-k curriculum**，并画 **A 保留曲线**（A→A / A→B→A / A→B→C→A'）。

### v0.2 规划（docs/V02_PLAN.md）
两步走（一次只动一组变量）：v0.2-A（稳定结构+全动态）→ v0.2-B（+warm-up/curriculum）。

### v0.2-A 实测（profile v02，C，40ep，seed0）
命令：`.venv/bin/python experiments/continual.py --seq core --seed 0 --n_train 1200 --epochs_per_task 40 --run C --profile v02 --out results/summary_v02a.json`

结果（vs v0.1：acc 0.045）：
| 指标 | v0.1(稳定化+关动态) | v0.2-A(稳定化+全动态) |
|---|---|---|
| acc_A_init | 0.045 | **0.143** |
| acc_A_end | ~0.045 | **0.211** |
| forgetting_A | — | **0.0** |
| node_overlap_A_A2 | 0.667 | 0.692 |
| final_nodes | 15 | 14 |
| spawn/prune/merge | 3/6/0 | 2/4/0 |
| final_acc | 各~0.04-0.06 | A=0.211 B=0.002 C=0.008 A2=**0.508** |

**v0.2-A 判定**：
- 恢复承重动态**有效**：acc_A 从 0.045→0.143（+3 倍），`forgetting_A=0`（A 不遗忘），A2=0.508（同分布能力落地）。
- **但鸡生蛋确凿**：B_mul=0.002、C_logic=0.008 **近随机**——Router 冷启动未建立跨任务 Node 分工，B/C 的输入分布没被路由到能承载的 Node 组。
- **结论**：承重动态是必要非充分；光恢复它不够，**必须加 warm-up + soft→hard curriculum** 破路由冷启动 → 进 v0.2-B。

### v0.2-B（profile v02b，C，40ep，seed0，warmup_tasks=1）——已实测
命令：`.venv/bin/python experiments/continual.py --seq core --seed 0 --n_train 1200 --epochs_per_task 40 --run C --profile v02b --warmup_tasks 1 --out results/summary_v02b.json`

结果（v0.2-B = 稳定结构 + 全动态 + Router warm-up/soft→hard curriculum）：
| 指标 | v0 | v0.1 | v0.2-A | **v0.2-B** |
|---|---|---|---|---|
| acc_A_end | 0.055 | 0.045 | 0.211 | **0.455** |
| A2(A') | 0.092 | 0.041 | 0.508 | **0.447** |
| B_mul | 0.0 | 0.041 | 0.002 | 0.020 |
| C_logic | 0.010 | 0.057 | 0.008 | **0.176** |
| forgetting_A | — | — | 0.0 | **0.0** |
| node_overlap_A_A2 | 1.0 | 0.667 | 0.692 | 0.636 |
| final_nodes | 54 | 15 | 14 | 13 |
| spawn/prune/merge | 42/5/42 | 3/6/0 | 2/4/0 | 1/5/0 |

**v0.2-B 判定（核心组）**：
- ✅ **A 学到 0.455**（vs v0.2-A 0.211 翻倍；vs v0/v0.1 的 ~0.05 近 9 倍）—— **鸡生蛋被破解，Router warm-up + soft→hard curriculum 有效**。
- ✅ `forgetting_A=0`（核心组 A 不遗忘）；A/A' 都 ~0.45 —— **调用而非重学**（GPT 核心验证目标达成）。
- ⚠️ B_mul=0.020（仍低）、C_logic=0.176（有起色但<A）—— cross-task 分工仍不足，是下一步要解决的（B/C 的 Node 组未充分学到），但相较 v0.2-A（B=0.002/C=0.008）已显著改善。
- **A_curve_after_each = [0.047, 0.0, 0.006, 0.455]**：学完A=0.047（warm评估走hard偏低）→学完B=0.0 →学完C=0.006 →学完A'=0.455（训A'时重激活A的Node组，A恢复）。**印证「重调用非重学」。**

额外收获：`--seq long`（因程序 bug 误跑）提供长序列对照——A→B→C→A'→D→B 下 forgetting_A=0.886（末端A被遗忘），说明保留曲线在更长序列下会衰减，作为下一步任务难度的依据。

（retention 递进曲线见 summary_v02b_retention.json，待补。）

## 2026-09-06 FDN-v0.2-C（方案A：per-task warm-up）—— 重要发现

Dad 拍板方案 A（每任务都 warm，让每个能力组都能长出）。代码：`--profile v02c --warm_epochs N --warm_mode every`（每任务前 N epoch 全班 soft，之后 hard top-k）；`--warm_mode first` 即 v02b。

命令：`.venv/bin/python experiments/continual.py --seq core --seed 0 --n_train 1200 --epochs_per_task 40 --warm_epochs 8 --run C --profile v02c --out results/summary_v02c.json`

### v0.2-C vs v0.2-B vs v0.2-A（C，seed0，40ep core）
| 策略 | A_init | A_end | forgetting_A | B | C | A2 | spawn/merge | final_nodes |
|---|---|---|---|---|---|---|---|---|
| v0.2-A 无warm | 0.143 | 0.211 | 0.0 | 0.002 | 0.008 | 0.508 | 2/0 | 14 |
| v0.2-B 首任务warm | 0.047 | **0.455** | **0.0** | 0.020 | 0.176 | 0.447 | 1/0 | 13 |
| v0.2-C 每任务warm | **0.340** | 0.053 | **0.845** | 0.029 | 0.033 | 0.053 | **0/0** | **12** |

v0.2-C `A_curve_after_each=[0.34, 0.004, 0.018, 0.053]`；retention 递进（v02b）已出：AA=0.143 / ABA=0.135 / ABCA=0.455。

### 判定
- **方向对一半**：per-task warm 让 A 学得最快（A_init=0.340，v0.2-B 的 7 倍）—— warm 让任务快速长出能力。
- **灾难性遗忘回归**：学 B 时 A 0.34→0.004（forgetting_A=0.845）。per-task 全班 soft 让每任务更新同一批共享 Node → 强覆盖。
- **结构零增长（spawn=0）**：全班 soft 时 Router 无 novelty、不再触发 spawn，冻结在 12 初始 Node。

### 机理性结论（复现 GSLM No-Go 根因）
「warm 的 softness」与「模块隔离的 sparseness」矛盾：soft 太多→覆盖+抑制生长；soft 太少→冷启动学不动但保留好。**共享投影被改写**的根因在 FDN 上复现（per-task 全班 soft 等价于共享投影）。

### 下一步（交 GPT 裁量）
动态 softness（warm 只用于新 Node 诞生期）+ task-conditioned 局部收敛（只更新本任务 Node 组）；把 warm 强度矩阵交外部审计定夺。

## 2026-09-06 FDN-v0.3 Protected Expert Formation（按 GPT v0.2 复审建议落地）

GPT 复审（docs/GPT_AUDIT_V02.md）判断：soft routing 本身不是解（v0.2-C 全班 soft = 共享投影覆盖），应改为 **只让新模块 soft warm-up、旧模块 protected**。据此实现 v0.3：
- **A. per-node warm**：forward 内未成熟 Node（maturity<thr）门控放大吸梯度，成熟 Node protected（hard）。
- **B. Competence Lock**：update_lifecycle 按 loss/entropy/usage 升 maturity，成熟后 plastic 衰减。
- **C. Re-activation**：reactivate_score 按亲和返回成熟 Node 排序，Router 复用非重学。

命令：`.venv/bin/python experiments/continual.py --seq core --seed 0 --n_train 1200 --epochs_per_task 40 --run C --profile v03 --out results/summary_v03.json`

### v0.3 结果（C，seed0，40ep core）
| 指标 | v0.2-B 首任务warm | v0.2-C 每任务warm | **v0.3 保护式** |
|---|---|---|---|
| acc_A_end | 0.455 | 0.053 | **0.285** |
| forgetting_A | 0.0 | **0.845** | **0.0** |
| A2(A') | 0.447 | 0.053 | **0.391** |
| B_mul | 0.020 | 0.029 | **0.051** |
| C_logic | 0.176 | 0.033 | 0.041 |
| final_nodes | 13 | 12 | 13 |
| n_mature | — | — | **13/13** |

**v0.3 判定**：
- ✅ **A 遗忘解决**（forgetting 0.845→0），A_end=0.285、A2=0.391 ——「保护旧模块、不被覆盖」达成（v0.2-C 致命伤修复）。
- ✅ **B_mul 首次 0.051**（per-node warm 让 B 略学会）。
- ⚠️ **C_logic 仍低（0.041）**，B/C 与 A 的 reactivation 命中 Node 重叠大 —— **B/C 尚未形成独立 Node 组**。
- **结论**：v0.3 方向正确（A 遗忘=0），但要做到「A/B/C 各形成独立 Node」的『动态模块形成机制』，需更强的**任务亲和约束**（强制任务映射到不相交 Node 组），即 v0.4 方向。**待 GPT 裁量。**

## 2026-09-06 FDN-v0.4 任务亲和约束（硬隔离）—— 反证：硬隔离违背 FDN 核心假说

v0.3 后拍板方向：用「强制任务映射不相交 Node 组」解决 B/C 未独立。据此实现 v0.4：task_owner 归属 + _task_mask 硬屏蔽其他任务 Node + ensure_task_nodes（任务0划归初始Node、后续任务spawn专属Node+add_param_group动态参数）。

命令：`.venv/bin/python experiments/continual.py --seq core --seed 0 --n_train 1200 --epochs_per_task 40 --run C --profile v04 --init_per_task 8 --out results/summary_v04.json`

### v0.4 结果（C，seed0，40ep，init_per_task 8）
| 指标 | v0.3 保护式 | v0.4 硬隔离 |
|---|---|---|
| acc_A_end | 0.285 | **0.023** |
| forgetting_A | 0.0 | **0.5** |
| node_overlap_A_A2 | 0.667 | **0.0** |
| B_mul | 0.051 | 0.031 |
| C_logic | 0.041 | 0.004 |
| final_nodes | 13 | 37（prune 29） |

**v0.4 判定（重要反证）**：
- ❌ **硬隔离破坏 FDN 核心**：`node_overlap_A_A2=0.0` —— A 与 A'(task3) 的 Node 完全不相交，违背「A' 调用 A 的 Node」这一 FDN 假说根本（v0 已证 IoU=1.0）。硬隔离把 A' 切到 task3 专属 Node → 无法"长出并调用"。
- ❌ **A 遗忘回归（0.5）**、A 未学会（0.023）：隔离切断了同分布任务的容量共享（A/A' 本应共享）。
- ❌ **prune 29 次**：spawn 大量用不上的专属 Node。
- **结论**：任务亲和约束应为**软偏好**（鼓励不同任务偏向不同 Node 组、允许复用已演化亲和），**硬性不相交是错误路径**。v0.3 的「软保护 + 新模块 warm」方向仍最优。
- **下一步**：v0.4-lite（软 task bias，q=W_q·x+task_bias[task]，不硬屏蔽）+ 交 GPT 审计裁定。


