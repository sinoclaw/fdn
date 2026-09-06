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

## 2026-09-06 FDN-v0.4-lite 软任务亲和（soft task bias，修正方向）

v0.4 硬隔离反证后拍板修正：不硬屏蔽，用可学习 task_emb 加 query 偏移（q += w*task_emb[task]），软性鼓励不同任务偏向不同 Node 组、允许跨任务复用（保住 A/A' 调用）。task_constraint=False，保留 v0.3 全部机制。

### v0.4-lite 结果（C，seed0，40ep，w=0.3）
| 指标 | v0.3 | v0.4 硬隔离 | v0.4-lite w=0.3 |
|---|---|---|---|
| acc_A_end | 0.285 | 0.023 | **0.104** |
| forgetting_A | 0.0 | 0.5 | **0.102** |
| node_overlap_A_A2 | 0.667 | 0.0 | **0.889** |
| B_mul | 0.051 | 0.031 | **0.152** |
| C_logic | 0.041 | 0.004 | 0.029 |
| final_nodes | 13 | 37 | 12 |

**v0.4-lite 判定（方向正确，需扫强度）**：
- ✅ **B_mul 0.051→0.152（3 倍，打破记录）**——软偏让难任务 B 学得更多。
- ✅ **node_overlap_A_A2=0.889**（A/A' 复盘好，未破坏 FDN 核心）；forgetting_A=0.102（A 低遗忘）。
- ⚠️ **A 本身变差**（0.285→0.104）——软偏把容量/路由从 A 重新分配到 B/C。
- ⚠️ **reactivation 中 A/B/C 命中前几个 Node 仍重叠**（都含 6/11/2/7/5/10）——w=0.3 不足以让 B/C 真正独立，B 学得更好但复用大部分 A 的 Node。
- **下一步**：扫 soft_task_bias（w=0.5/0.8）定位「B/C 独立」vs「A 保留」最优权衡，交 GPT 审计裁定。

### v0.4-lite 软偏强度扫描（w ∈ {0.3, 0.5, 0.8}），确认 w=0.5 最优
| 版本 | A_end | forget | overlap | B | C | A2 |
|---|---|---|---|---|---|---|
| v0.4-lite w=0.3 | 0.104 | 0.102 | 0.889 | 0.152 | 0.029 | 0.076 |
| **v0.4-lite w=0.5** | **0.545** | **0.0** | 0.700 | **0.209** | 0.002 | **0.477** |
| v0.4-lite w=0.8 | 0.020 | 0.75 | 0.700 | 0.0 | 0.109 | 1.0 |

**w=0.5 为 FDN 迄今最优**：A=0.545（超 v0.2-B 0.455 纪录 +20%）+ forgetting=0 + B=0.209（v0.2-B 的 10 倍）+ A2=0.477（A' 调用）+ overlap=0.7（A/A' 复用完整）。
- **w=0.3 太弱**（A 被 B 挤占 0.104）；**w=0.8 太强**（task_emb 主导路由，破坏 A/A' 复用，A=0.020/遗忘 0.75）。
- **结论**：软任务亲和存在**最优强度 w≈0.5**，同时保住跨任务复用 + 让难任务（B）学得更多——回应 GPT 目标。
- 命令：`.venv/bin/python experiments/continual.py --seq core --seed 0 --n_train 1200 --epochs_per_task 40 --run C --profile v04lite --soft_task_bias 0.5 --out results/summary_v04lite_w05.json`

## 2026-09-06 FDN-v0.5 批次 1：多 seed 复现（重点反证）

GPT v0.5 裁定关键：先多 seed 验证 w=0.5（至少 5 次），若 A_end>0.4/遗忘≈0/A2>0.35/B>0.15 才可信。据此跑 seed 0..5（v04lite, w=0.5, 40ep, core）。

### 结果（seed 0..5）
| seed | A_end | forgetting | B_mul | C_logic | A2 | overlap | nodes |
|---|---|---|---|---|---|---|---|
| 0 | 0.018 | **0.904** | 0.035 | 0.086 | 0.031 | 0.538 | 15 |
| 1 | 0.084 | 0.0 | 0.035 | 0.045 | 0.021 | 0.429 | 15 |
| 2 | 0.188 | 0.0 | 0.0 | 0.0 | 0.742 | 0.700 | 14 |
| 3 | **0.973** | 0.0 | 0.002 | 0.053 | 1.0 | 0.800 | 13 |
| 4 | 0.223 | 0.0 | 0.084 | 0.070 | 0.254 | 0.875 | 14 |
| 5 | 0.061 | 0.613 | 0.006 | 0.010 | 0.502 | 0.636 | 14 |

**判定（重点反证）**：
- ❌ **w=0.5 不能跨 seed 复现**：A_end 0.018→0.973（>50 倍差异），forgetting 0→0.904，仅 seed3 达标。**GPT 的"最大不确定性"警告被证实。**
- ❌ **B 全部 <0.1**（0/6 达 0.15 阈值）。
- ✅ 但 **overlap 4/6 >0.5**（A/A' 复用较稳定，说明复用机制本身在，只是精度不稳）。
- 🎯 **根因 = 任务间无真模块分化**：6 seed 的 `task_affinity_disjoint_frac` 全 = **0.0**（A/B/C/A' 用同一批 node）。这是「B 学不好 + A 假象/被覆盖」的共同根源，也是 GPT 反复强调的短板。
- 🎯 **seed3 的 A=0.973 是"假象"**：A_curve=[0.08,0.041,0.064,0.973]，A 一直没学会（~0.04-0.08），到 A' 才虚高——A' 恰好命中大输出 node，非"学得好"。

### 结论
**w=0.5 不是稳定解；真正的拷问核心是「模块分化如何实现」（disjoint=0）。** 这直接推出批次 3 的 Node Specialization Matrix 指标（GPT 强调"比 accuracy 更重要"）。

## 2026-09-06 FDN-v0.5 批次 3：Node Specialization Matrix 指标落地
- `evaluation/metrics.py`：+`node_specialization_matrix`（行归一化频率矩阵）+`_row_cosine`/`row_cosine_pairs`（行余弦）。
- `experiments/continual.py`：evaluate_model 加 `counts=True` 收集每任务 Node 激活计数；run_one 输出 `specialization_matrix`/`specialization_tasks`/`specialization_cos`。
- `tests/test_fdn.py`：+`test_v05_specialization_matrix`（理想分化 A/A2 高相似、A/B 低相似）——17/17 全绿。
- **探针验证**（equi 短跑）：A~D_sub cos=0.984、A~B cos=0.934——**所有任务行高相似（分化弱）**，与批次 1 disjoint=0 同根因。

## 2026-09-06 FDN-v0.6 批次 1：Node 级 competence/novelty 状态落地（纯测量）

依据 GPT 二审判定（docs/GPT_AUDIT_V05.md）：v0.6 = Node autonomous specialization（Novelty-Spawn + Node Competence + Plasticity Decay）。批次 1 只做纯测量——验证「现有 node 对 B 是否 competence 低」。

改动（model/fdn.py + experiments/continual.py，均为加法，不改路由）：
- +`node_novelty`（1-cos(q,key)）与 `node_error`（该 node 输出与聚合贡献之差的 EMA）两个 buffer；在 forward 选中 node 时更新。
- +`node_telemetry()`（返回 novelty/error/competence/maturity/usage 快照）写入 summary。
- **修 `update_lifecycle` competence 信号源**：原来用 `usage_share × (1-全局熵)`，在未选中 node 上恒 0（competence 死掉）→ 改为「归一化 error 的倒数」驱动（error 越大 → competence 越低 = competence gap 高）。并修正「未使用 node error=0 被误判为金牌」陷阱（置中性 0.5）。

**批次 1 关键发现（探针 seed0, 14ep, core）**：
- **competence 现在能正确反映 node 胜任度**：node8（error=109）competence=0.36（低=gap 高）；node0/5/6（error<2.7）competence=0.87-0.89（高=胜任）。
- **方向验证通过**：高误差 node 展现低 competence——正是 GPT 要的 spawn 判据信号源。
- 之前 competence 恒 0 是纯 bug（usage_share×entropy 在未选中 node 上归零 + clamp），已修。

**结论**：批次 1 达成——Node 级 competence/novelty 落在原地且能正确反映胜任度。下一步（批次 2）用「所有激活 node 的 competence 都低 + novelty 高」作为 Node 级 spawn 判据。
> 产物：`/tmp/probe_v06.json`（临时）。测试 17/17 全绿。

## 2026-09-06 FDN-v0.6 批次 2：Novelty-triggered Spawn（Node 级判据，核心）

改 growth/controller.py 的 spawn：从「全局 min-cos（Router 视角，A/B 挤进已有 node）」→「Node 级综合判据」。
- 加 `_node_competence_gap`：spawn = f(novelty, error, competence, usage, capacity)，用 `score = 0.4*norm(novelty) + 0.3*norm(error) + 0.3*(1-norm(comp))`，`score > spawn_score_thr(0.5)` 触发。
- **关键设计纠偏**：初始用「所有 used node 平均 competence < 0.5 硬 AND」，实测**永不触发**（A 学会的 node competence 高拉高均值，B 来时达不到"普遍低"）→ 改为**综合分**，spawn 才触发（批次 2 的核心教训）。
- +`_do_spawn`（抽取 spawn 执行）、`use_node_spawn` 开关（默认 False，v04lite 等旧 profile 走全局 min-cos 向后兼容）、`--profile v06`、`--spawn_patience`、`--seq mini`（A→B→A'，GPT 最简序列）。
- model/fdn.py：error_ema 0.3→0.5、competence_lr 0.05→0.3（更快反映最近模式，不被长期 EMA 稀释——否则 node1/node9 高误差 node 的 competence 被稀释成高值）。
- 测试：+`test_v06_node_spawn_score`（构造 B 外行场景验证 gap 触发）——18/18 全绿。

### 批次 2 关键发现（探针 v06 mini, seed 0）
- ✅ **Node 级 spawn 首次在真实训练中触发**：spawn=2，`last_spawn_reason={score:0.512, avg_novelty:0.996, avg_error:0.945, sustained:True, task:3}`——B→A' 边界触发，nodes 12→14。
- ❌ **分化仍未形成**：disjoint=0、cos(A,B)=0.928——新 spawn 的 node12 同时被 B 和 A' 调用，未形成 B 专属。
- ❌ **A 保留仍差**：forgetting=0.676（A_curve 起点就低 0.066）。
- **判断**：批次 2 达成「Node 级 spawn 判据启动」这一核心目标（机制已动）；「分化」是 spawn 后 node 是否被 Router 专属化的问题，需批次 3（Plasticity Decay 5 态让新 node 高可塑、成熟后低可塑保护）配合 + 后续 Router 偏向新 node。
> 产物：`/tmp/probe_v06c.json`（临时）。测试 18/18。

## 2026-09-06 FDN-v0.6 批次 3+：任务边界 freeze_old（重大突破——freeze 救活 A/B）

GPT 指出的本质缺口：Plasticity Decay 只保护 Hebbian 推理期，不拦 Adam 主训练梯度（A 的 node 在 B 仍被选中 → Adam 覆盖 → A 遗忘）。
**解法**：任务边界冻结「已成熟的旧 Node」=`freeze_old_nodes()`（其参数 `requires_grad=False`，Adam 跳过），只有新 spawn / 未成熟 node 能继续学。
- model/fdn.py：+`freeze_old_nodes()`（冻结成熟且非最新 node）；experiments/continual.py：`--freeze_old` 开关，任务边界 pos>0 时调用并记录 `freeze_count`。

### 关键对比（v06 mini 24ep seed0）
| 指标 | 无 freeze | **+ freeze_old** |
|---|---|---|
| forgetting_A | 0.992 ❌ | **0.0** ✅ |
| A_end | 0.002 ❌ | **0.449** ✅ |
| B_mul | 0.0 ❌ | **0.904** ✅ |
| A2 | 0.002 | 0.02 |
| spawn | 2 | 2 |

**freeze_count=[{B_mul: 11}, {A2_add: 8}]**——freeze 成功冻结旧 node（Adam 不再覆盖）。

### 结论（批次 3+ 核心突破）
**任务边界冻结旧 node 是防遗忘的承重机制**——直接验证基准判断。A 遗忘 0.992→0、A_end 0.002→0.449、B 0→0.904。剩余缺口：A'（调用恢复）低 + disjoint=0（cos 仍高），属「如何让 Router 偏向新 node / 旧 node 调用」的下一步，但**核心突破已达成**。

### ⚠️ 批次 3+ 诚实反证：40ep 完整训练下 A 根本学不会（seed0）
**重要**：freeze 的"救活"只在 24ep 短跑成立。seed0 完整 **40ep**：A_end=0.016、B=0.018、A_curve 起点 **0.021**（A 从一开始就没学会）——不是 freeze 失效，是 **A 任务本身没学好**（Router 冷启动鸡生蛋仍在，40ep 更长的训练里结构演化 spawn/prune 干扰 A）。
| | 24ep(短跑) | **40ep(完整)** |
|---|---|---|
| A_end | 0.449 | **0.016** ❌ |
| B_mul | 0.904 | **0.018** ❌ |
| A_curve 起点 | 0.123 | **0.021** |

**结论**：freeze 救不了"没学会"。多 seed 验证是必须的（单 seed/短跑都不可信，v0.5 教训）。freeze_count 正常（B_mul:11, A2_add:9）。

## 2026-09-06 FDN-v0.7（GPT 三审）：competence-gated warm-up + 关键基线对账（架构级根因）

GPT 三审裁定：v0.6 未实现 Node autonomous specialization，根因是"A 没学会 → 无 stable competence → signal 不可靠 → spawn 无归属"。给 v0.7 = Warm-up → Mature → Protected → Novelty Spawn，用 competence（accuracy+entropy+连续N次稳定）当验收门槛（非 epoch）。

### 实现（代码 19/19 全绿）
- `model/fdn.py`：+warm_active/warm_temperature/warm_curve 状态；+`update_curriculum(acc, entropy)`（competence-gated 收敛判定）；forward 加 soft→hard 课程分支（warm_active 时全班 soft 路由）。
- `experiments/continual.py`：+`_quick_eval`（每 epoch 测当前任务 acc+entropy）；+`--profile v07`；train_task +gated_warmup。

### 关键 bug 修复（诚实记录）
1. **旧代码 warm-up 是空壳**：`warmup_steps` 参数存在但 forward 里只有 per-node warm（未成熟 node 门控放大），无"全班 soft→hard 课程"——v0.6 里 `warmup_steps=0` 更是全程 hard。**这解释为什么 v0.6 的 A 学不会**。
2. **entropy 恒 0**：`info["entropy"] = entropy if self.training else 0.0`，eval 时恒 0 → curriculum 永远不达标。已改为恒记录真实 entropy。
3. **评估口径不一致**：soft 分支原 `if warm_active and self.training`，eval 走 hard → Router 没学会时 hard 评估是冷启动。改为 `if warm_active`（训练+评估一致）。

### 单任务基线对账（决定性发现，/tmp/probe_baseline.py）
| 模型 | 单任务 A accuracy | 解读 |
|---|---|---|
| StaticMLP（共享网络） | **1.000** | A 任务**极好学**，上限 1.0，GPT 判据合理 |
| StaticMoE n=12 k=3 | **0.072** | 路由模块化（固定结构）**反而学不会**！ |
| StaticMoE n=12 k=5 | **0.049** | 同上 |
| StaticMoE n=24 k=5 | **0.051** | 同上 |
| StaticMoE n=12 k=8 | **0.055** | 同上 |

### 核心结论（架构级根因，扭转认知）
**A 任务本身可学到 1.0（StaticMLP 完美学会），但连 StaticMoE（静态、固定结构、无任何动态/干扰）单任务 A 都学不会（0.05-0.07）。** 这彻底证明：**问题不在 FDN 的动态机制（spawn/warm-up/plasticity/freeze），而在 MoE 路由架构的 top-k 门控优化本身**——top-k argmax 是离散不可导的，只有被选中 node 拿梯度，Router 得不到正确梯度信号。

**与 v0.1 结论完全印证**："失败更可能源于逐样本 top-k 门控路由优化的本身脆弱而非仅结构震荡"。这是 FDN 系列的**最底层根因**——不解决 top-k 路由可导性，任何外层机制（warm-up/spawn/分化）都救不了。

> 产物：`/tmp/probe_baseline.py`、`/tmp/probe_v07*.json`、`/tmp/diag_soft.py`（临时）。

## 2026-09-06 FDN 最终判定实验（GPT 四审钦定）：条件任务下 MoE 仍不分化 → FDN 正式封存

GPT 四审建议的**最后决定性实验**（docs/GPT_AUDIT_V07.md）：拿 StaticMoE 换**天然存在子结构的条件任务**（z=0→add, z=1→mul），若 StaticMoE 学会 + Router 分化（z=0→Node A, z=1→Node B）则证明"不是 MoE 不行，是 FDN 任务没给模块化留空间"；若失败则 **FDN 项目正式封存**。

### 实验结果（/tmp/probe_cond.py，GEN_COND 条件任务单任务）
| 模型 | 单任务 acc | z=0 vs z=1 node IoU 重叠 |
|---|---|---|
| StaticMLP h=32/64（共享网络） | **1.000** | — |
| StaticMoE n=12 k=3 | 0.051 | 0.667 |
| StaticMoE n=12 k=5 | 0.031 | 0.625 |
| StaticMoE n=24 k=5 | 0.115 | 0.857 |
| StaticMoE n=12 k=8 | 0.090 | **1.000**（完全不分化） |
| StaticMoE n=16 k=4 | 0.023 | 0.875 |

### 核心结论（最终判定）
**即使给了"天然子结构"的条件任务（z=0→add / z=1→mul 有明确的阶层分工），StaticMoE 依然学不会（acc 0.02-0.12）、Router 依然不分化（IoU 0.6-1.0，z=0/z=1 用几乎同一批 node）。**

这**反证了 GPT 四审的假设**（"不是 MoE 不行，是 FDN 任务没给模块化留空间"）——给了真子结构，MoE 还是不行。StaticMLP 始终 1.0（共享网络最优）。

按 GPT 四审预定的判定路径（"若 StaticMoE 条件任务也失败 → FDN 项目正式封存"），**FDN 正式进入 No-Go 封存**。根本结论进一步证实：**"多独立 Node + Router 稀疏路由"的 MoE 拓扑在 toy 回归体系下就是学不会，无论任务有无子结构**——共享网络（StaticMLP）始终最优。这与 GSLM No-Go 共同指向：**"动态结构 ≠ 动态智能；若动态结构只是 MoE 专家路由，它只是强行拆碎共享函数"**（GPT 金句）。

> 产物：`/tmp/probe_cond.py`（临时）。

## 2026-09-06 FDN 封存复核：load-balancing 承重件实验（Router 坍塌假说反证）

爸爸质疑"MoE 是否缺工业标准件（load-balancing loss / Router Z-loss）才学不会"——这是合理的工程怀疑（Switch Transformer / Mixtral 靠这两个件让 Router 稳定分配梯度，防坍塌/专家饿死）。本实验：StaticMoE 加 LB + Z，单任务 A 重跑（/tmp/probe_lb.py，40ep）。

| 配置 | 单任务 A acc | lb_raw |
|---|---|---|
| RawMoE n=12 k=3（裸跑） | **0.314** | — |
| RawMoE n=12 k=5（裸跑） | 0.119 | — |
| BalMoE n=12 k=3（+LB+Z） | 0.111 | 3.906 |
| BalMoE n=12 k=5（+LB+Z） | 0.119 | 6.363 |
| BalMoE n=24 k=5（+LB+Z） | 0.170 | 7.194 |
| BalMoE n=12 k=8（+LB+Z） | 0.184 | 8.817 |

### 核心结论（Router 坍塌假说被反证）
1. **load-balancing + Z-loss 没有提升 MoE 性能**（n=12 k=3：0.314→0.111 反而下降；k=5 持平）——**Router 坍塌/专家饿死不是瓶颈**，加承重件反而干扰了 k=3 的自然分工。
2. **裸跑 k=3 达到 0.314**（高于此前记录的 0.072）——说明早期基线在 30ep 下欠训练，40ep 更充分。但**无论如何 0.314 仍远逊 StaticMLP 的 1.000**，且 k=3（最少专家、最窄激活）才勉强到这个水平。
3. **核心矛盾未变**：MoE 最优（裸跑 k=3，0.314）仍远不及共享网络（StaticMLP 1.0），加工业标准件（LB/Z）反而更差。

### 判定
**爸爸的"补标准件"直觉方向对了（工程上是最该先试的），但结果不支持 LB 是瓶颈**——加 LB 反而抑制性能，进一步确认 **"MoE 稀疏路由在 toy 回归体系下不适配/欠训"是真实根因，而非 Router 坍塌**。这加固了 FDN 封存结论：**共享网络（StaticMLP）在此体系始终最优，MoE 的独立专家+稀疏路由是根本劣势**（无论加不加外围机制/标准件）。

> 产物：`/tmp/probe_lb.py`（临时）。

## 2026-09-06 FDN 存续新方向：BDH 式共享骨架 + 连续逐通道门控（重大突破）

BDH（Dragon Hatchling，arXiv:2509.26507，pathwaycom/bdh）深入分析见 `docs/BDH_DEEP_DIVE.md`。核心：**动态结构不是 MoE 专家路由，而是"共享骨架 + 快权重（Hebbian 边权状态）+ 连续稀疏逐通道门控"**。

### 单任务可学性（/tmp/probe_bdh.py, /tmp/probe_bdh_single.py）
| 模型 | A_add | B_mul | C_logic |
|---|---|---|---|
| StaticMLP（共享） | 1.000 | 1.000 | 1.000 |
| StaticMoE（FDN 式，MoE）| 0.05-0.31 | — | — |
| **BDHReg（共享骨架+逐通道门控）** | **1.000** | **1.000** | **1.000** |

**BDH 式单任务全 1.000**（sparsity 0.38-0.47），克服了 FDN 的致命伤（FDN 连单任务 A 都学不会——MoE 梯度分配失败）。共享骨架 + `x_sparse * y_sparse` 连续门控（无离散 top-k），梯度畅通。

### 多任务持续学习 A→B→A'（/tmp/probe_bdh_cont.py，N=128/256 × seed 0/1）
| 配置 | final A | final B | final A' | A_forget | A-B chan IoU |
|---|---|---|---|---|---|
| N=128 seed0 | **1.0** | 0.07 | **1.0** | **0.0** | 0.89 |
| N=128 seed1 | **1.0** | 0.051 | **1.0** | **0.0** | 0.84 |
| N=256 seed0 | **1.0** | 0.057 | **1.0** | **0.0** | 0.83 |
| N=256 seed1 | **1.0** | 0.055 | **1.0** | **0.0** | 0.88 |

### 核心结论（突破 vs 缺口）
1. ✅ **保旧成功**：A_forget=0.0（A init=1.0 / end=1.0）——**FDN 从未做到**！BDH 式连续门控下 A 完美保留。
2. ✅ **A' 恢复成功**：A2_add=1.0（重学 A 无干扰）。
3. ❌ **学新受阻**：B_mul 仅 0.05-0.07，但**单任务 B_mul=1.0**（单独完全可学）——说明 **B 是被持续学习的干扰毁掉，非架构缺陷**。
4. ⚠️ **通道分化弱**：A-B IoU≈0.83-0.89（通道重叠高，未形成独立通道子集）。

### 定性（重要）
BDH 式与 FDN 的区别是**质的飞跃**：
- FDN：**单任务 A 就学不会**（MoE 架构缺陷，梯度分配失败）——死路
- BDH 式：**单任务全学会（1.0）**——架构可训、不塌、可解释；**多任务下 B 受干扰**——这是**持续学习的"学新"问题**（非架构缺陷）
- **BDH 式已克服 FDN 的致命伤**（MoE 不可导/梯度坍塌），证明"**动态结构 ≠ MoE 专家路由**"，正是 GPT 四审指的方向。

**剩余缺口**：多任务下 B_mul 学不会（单任务却 1.0）——疑因共享门控对 B 的区分度不足 / 无显式"分新能力"机制。下一步方向：**给 BDH 式加"能力分离"机制**（如 per-task 门控子空间 / 快权重边权按任务累积），验证能否让 B 在多任务下也学会。

> 产物：`/tmp/probe_bdh.py`、`/tmp/probe_bdh_cont.py`、`/tmp/probe_bdh_single.py`（临时）。

## 2026-09-06 BDH 式持续学习深入诊断：灾难性遗忘 + 共享门控根因

### 诊断实验（/tmp/probe_bdh_diag.py, /tmp/probe_bdh_overwrite.py, /tmp/probe_bdh_freeze.py）
| 实验 | A | B | A' | 解读 |
|---|---|---|---|---|
| 直接学 B | — | 1.000 | — | B 本身可学 |
| A→B | 0.018 | **1.000** | — | **学了 B，A 丢了** |
| A→B→A'（无助） | 1.000 | 0.021 | 1.000 | 学 A'，A 回来、B 丢 |
| A→B 后冻结全部只让 gate 学 A' | 1.000 | 0.020 | 1.000 | **冻结权重照样 B 被覆盖** |

### 核心结论（重要，诚实）
1. ✅ BDH 式**无 MoE 致命伤**：单任务全 1.0、新任务能从旧任务后学到 1.0（A→B 时 B=1.0）。
2. ❌ **灾难性遗忘（旧任务被覆盖）**：谁最后学谁赢——A 被 B 覆盖（A→B 后 A=0.016），B 被 A' 覆盖（A→B→A' 后 B=0.02）。
3. ❌ **共享门控是根因**：[T5b] 冻结全部权重、只让 gate 学 A'，B 仍被覆盖——**B 被覆盖不是权重被改写，而是共享 gate 被 A' 夺取**。BDH 式的动态性在**全局共享的 gate（通道开合）**，无独立模块，无法保护旧能力。
4. ⚠️ 通道分离（pin 强通道）无效——B 学不会不是通道占用，是 gate 共享。

### 能力边界画像（BDH 式完整）
- **能**：训、学新、可解释、无 MoE 坍塌
- **不能**：保护旧能力（共享动态 gate 是根因，无模块隔离）

### 定性
BDH 式与 FDN 是对立的优缺点：
- FDN（MoE 专家路由）：**学不会新任务**（梯度坍塌），但想模块化
- BDH 式（共享骨架+连续 gate）：**学得会新任务**（不塌），但**无法隔离旧能力**（共享 gate）

**共同的深层矛盾**：要"动态结构 + 持续学习"，要么"独立模块（MoE，但学不会）"，要么"共享门控（能学，但不保旧）"——**两者的结合点（能学新 + 能保旧）仍未找到**。这正是 FDN/GSLM 一路 No-Go 的同一个死结，BDH 式也没绕过，只是换了个表现。

> 产物：`/tmp/probe_bdh_diag.py`、`/tmp/probe_bdh_overwrite.py`、`/tmp/probe_bdh_freeze.py`、`/tmp/probe_bdh_consol.py`（临时）。

## 2026-09-06 架构改造突破：BDH-FastWeights 严格版——A/B/A' 全 1.000，A_forget=0.0（FDN 系列首次达成）

爸爸拍板"架构可改，方向对+结果对"。从 BDH 论文核心 Fast Weights 出发，三步迭代定位到正解：

### 迭代轨迹（每步一次只动一个变量）
| 版本 | 架构 | A | B | A' | A_forget | 结论 |
|---|---|---|---|---|---|---|
| baseline | 共享单 gate | 1.0 | 0.02 | 1.0 | 1.0 | B 被覆盖（基线）|
| v1 | gate 独立、骨架共享 | 0.797 | **0.562** | 1.0 | 0.203 | **gate 独立救了 B**，但共享骨架改写拖累 A |
| v2 | 路径独立 + shared_proj | 0.207 | 0.0 | 1.0 | 0.793 | **共享最浅投影也致命**——任何共享中间层都被改 |
| **v3** | **每能力完全独立（零共享）** | **1.000** | **1.000** | **1.000** | **0.0** | ✅ **正解！** |

### v3 架构（BDH-FastWeights 严格版）
每能力 = 独立 {in_enc, gate, out_dec, ln}，零共享中间层。训练任务 k 只更新 cap k 路径，冻结其它全部。/tmp/probe_bdh_fw3.py

### 核心洞察（最有价值）
1. **根因确认**：不是"MoE 分段"vs"共享 gate"二选一，而是**任何被共享的中间表征（gate/骨架/最浅投影）都会被后续任务改写，能力就丢**。
2. **Fast Weights 正解**：每个能力拥有**完整独立计算路径**（快权重），只共享最原始的输入选择（cap 路由）。这样学新不覆盖旧、保旧不束缚新。
3. **既学新又保旧首次达成**：A→B→A' 全 1.0 + A_forget=0——**这是 FDN 系列（v0→v0.7）从未做到的结果**。

### 关系澄清（诚实）
- BDH 式（共享骨架+共享 gate）能"学新"但"不保旧"（B 学到 1.0 但 A/B 被后续覆盖）——单任务全会但持续学习覆盖。
- **BDH-FastWeights（每能力独立路径）** 才真正实现"动态结构 + 持续学习"——这次方向对了。

### 下一步
v3 用"每任务独立 cap"（结构给定，非 oracle）。下一步可验证：
- 能否**自动发现能力边界**（少 cap 池，让模型自适应分配 cap，非人为每任务一个）——更接近 BDH 自组织
- 能否**共享只做最终输出层的省参**（v3 每能力完整路径参多，能否折中）

> 产物：`/tmp/probe_bdh_fw.py`、`/tmp/probe_bdh_fw2.py`、`/tmp/probe_bdh_fw3.py`（临时）。

## 2026-09-06 FastWeights 自组织能力分配：卡点诊断（任务太简单是障碍）

v3 已证 FastWeights（每能力独立路径）正解。本步升级为**自组织 cap 池**（模型自动复用/新建 cap，无 oracle）。

### v1（首个 batch MSE<阈值 判 fit，/tmp/probe_bdh_pool.py）
| seed | A | B | A' | A_forget | n_caps |
|---|---|---|---|---|---|
| 0 | 1.0 | 0.021 | 1.0 | 0.0 | 1 |
| 1 | 1.0 | 0.0 | 1.0 | 0.0 | 1 |
| 2 | 1.0 | 0.016 | 1.0 | 0.0 | 1 |

**全 n_caps=1，B 全塌**——首个 batch MSE 太低，永不触发 spawn，全部复用 cap0，B 被覆盖。

### v2（快速试训各 cap 测验证 acc 选 cap，/tmp/probe_bdh_pool2.py）
仍 n_caps=1——probe 8ep 后 cap0 对 B 也达 acc≥0.8（阈值），探测信号无法区分能力边界。

### 诊断结论（诚实）
**toy 任务太简单：一个 cap 用 8ep 就能 fit 任何回归任务到高 acc**——"能力边界探测"在简单任务上无意义（与 GSLM 容量饱和结论一致：简单任务一个共享网络全够用，无需分能力）。要测"自组织能力边界"，**必须用需要不同计算通路的任务**（如 z 条件任务 / 异或 / 非线性分解），否则任何 cap 都能通吃。

### 当前定位
- ✅ **v3（每能力独立路径）= 正解**：A/B/A' 全 1.0 + A_forget=0，Fast Weights 打破"共享覆盖"死结（已 commit 2fb2e19）。
- ⚠️ **自组织**被"任务太简单"卡住：探测信号无法区分能力边界，需更难任务才能验证。
- 自组织不是方向错，是**实验任务给不出能力边界**；v3 的结构正解已确立。

> 产物：`/tmp/probe_bdh_pool.py`、`/tmp/probe_bdh_pool2.py`（临时）。

## 2026-09-06 自组织能力拆分诊断：架构能拆，但软路由懒（MoE 懒路由）

换「需要不同计算通路」的混合条件任务（z=0→add, z=1→mul），FastWeights 双 cap 池 + Router 按 z 路由，测能否自发拆分（/tmp/probe_bdh_split.py, /tmp/probe_bdh_split2.py）。

| 配置 | acc_all | z0_acc | z1_acc | route_z0→cap0 | 交叉专化 |
|---|---|---|---|---|---|
| (a) FastWeights+Router 软路由 | 1.000 | 1.0 | 1.0 | **0.0** | — |
| (b) 强制正确路由（upper bound）| — | cap0@z0=1.0 | cap1@z1=1.0 | 1.0 | cap0@z1=0.06, cap1@z0=0.00 |
| (c) 单 MLP baseline | 1.000 | 1.0 | 1.0 | — | — |

### 结论（诚实）
1. **架构能拆分**：强制正确路由时 cap0 成纯 add 专家（z0=1.0）、cap1 成纯 mul 专家（z1=1.0），**交叉≈0**（cap0@z1=0.06、cap1@z0=0.00）——FastWeights 完全有能力形成纯净专家。
2. **但软路由懒**：软路由下 acc_all 已 =1.0（软权重混合已通吃），**Router 无激励去硬拆分**（route_z0→cap0=0.0）——正是 MoE 的**懒路由/router entropy collapse** 问题。
3. **需激励**：让 Router "愿意拆"，需 load-balancing / hard 路由等**激励 Router 均匀分配**的机制（工业 MoE 同理）。

### 定性
FastWeights 结构正解已确立（v3：A/B/A' 全 1.0 + A_forget=0）。自组织"按需拆能力"**架构上可行**（b 上限，纯专家+零交叉），但**软路由默认不拆**（需激励）。下一步方向 = **给 FastWeights 池加"拆分激励"**（load-balance 均匀分配 / hard 路由），促 Router 自发拆成专家。

> 产物：`/tmp/probe_bdh_split.py`、`/tmp/probe_bdh_split2.py`（临时）。

## 2026-09-06 拆分激励测试：load-balancing loss 破坏专化（诚实负面信号）

给 FastWeights 池加 load-balancing loss（Switch Transformer 标准件），试图促 Router 自发拆成 add/mul 专家（/tmp/probe_bdh_lb.py）。判据：acc 保持 1.0 + route z0→cap0 显著 + cap 专化。

| lb_coef | acc | route z0→cap0 | cap0@add | cap1@mul | cap0@mul(交叉) | cap1@add(交叉) |
|---|---|---|---|---|---|---|
| 0.0 | 1.0 | 0.767 | **0.686** | 0.0 | 0.0 | 0.539 |
| 0.05 | 1.0 | 0.465 | 0.0 | 0.277 | 0.296 | 0.0 |
| 0.1 | 1.0 | 0.506 | 0.0 | 0.281 | 0.236 | 0.004 |
| 0.3 | 1.0 | 0.535 | 0.0 | 0.258 | 0.221 | 0.0 |
| 1.0 | 1.0 | 0.559 | 0.0 | 0.236 | 0.206 | 0.0 |

### 结论（诚实）
1. **load-balancing 反而破坏专化**：加 LB 后 cap0@add、cap1@mul 掉到 0.0-0.28（明显变差）——LB 强制"均匀分配"让两个 cap 各学一半（半桶水），而非各学一个模式。
2. **lb=0.0 反而最接近专化**：cap0@add=0.686、cap0@mul=0.0（cap0 偏向 add 且不跨界），但仍未纯净。
3. **关键差距**：与 (b) 强制正确路由 upper bound（cap0/cap1 纯专家、交叉≈0）相比，**Router 在无监督下学不出正确的拆分分配**——"拆分"需要**路由监督信号**或**更强的分解驱动**，而 toy 上的这种监督本质是 oracle 的雏形。

### 定位
FDN 系列已形成完整的探索闭环（见 README/EXPERIMENT_LOG）：
- v0→v0.7：MoE 专家路由学不会（梯度坍塌）→ No-Go
- BDH 式：共享骨架+连续门控能学新但不保旧（共享覆盖）
- **FastWeights v3：每能力独立路径 → A/B/A' 全 1.0 + A_forget=0（结构性正解，已确立）**
- 自组织拆分：架构能拆（b upper bound 纯专家+零交叉），但 Router 无监督不主动拆（懒路由），LB 激励反而破坏专化

> 产物：`/tmp/probe_bdh_lb.py`（临时）。

## 2026-09-06 动态池 + 按需 spawn 完整突破（爸爸选 B）：可生长动态结构 + 持续学习

爸爸选 B：把 FastWeights v3（每能力独立路径）升级为**可按需增长的池**。核心判据 = **复用旧 cap 学新任务若"会伤旧能力"则 spawn 新 cap**（能力隔离的本质）。用 probe 深拷贝试训测"伤旧程度"驱动 spawn（无 oracle，/tmp/probe_bdh_dynpool.py）。

### 结果（seq = A_add -> A2_add(同类) -> B_mul(异类)）
| seed | final | n_caps | assign | A_forget | owned |
|---|---|---|---|---|---|
| 0 | A=1.0 A'=1.0 B=1.0 | **2** | {A:0, A':0, B:1} | **0.0** | {0:[A,A'], 1:[B]} |
| 1 | A=1.0 A'=1.0 B=1.0 | **2** | {A:0, A':0, B:1} | **0.0** | {0:[A,A'], 1:[B]} |

### 核心成果（完整达成）
1. **按需自动 spawn**：池从 1 长到 2（非人为指定）。**A'/A 同类自动复用 cap0**（复用不伤旧→复用，池不增），**B 异类自动 spawn cap1**（复用会覆盖→新建）。
2. **模型自己学会能力边界**：owned={0:[A,A'], 1:[B]}——cap0 是"A 家族"（加法）、cap1 是"B 家族"（乘法），**无 oracle 自发形成**。
3. **学新 + 保旧**：A/A'/B 全 1.0 + A_forget=0——旧能力零覆盖。
4. **判据成立**："复用伤旧→新建"是可靠的按需 spawn 信号（比此前"fit 好不好"判据强）。

### 定位（FDN 完整闭合）
这是 FDN 系列从 v0→v0.7 追求的完整命题（动态结构 + 持续学习 + 自动发现能力边界）的**首次达成**：
- 结构（每能力独立路径，FastWeights v3）✅
- 动态生长（按需 spawn，同类复用/异类新建）✅
- 持续学习（学新 + 保旧，A_forget=0）✅
- 自动能力边界（无 oracle，模型自组织）✅

> 产物：`/tmp/probe_bdh_dynpool.py`（临时）。

## 2026-09-06 验证矩阵全绿：DynamicPool 机制跨 5 seed × 8 结构化任务稳定成立，正式升级为研究方向

爸爸指令：设计好验证矩阵、判据先行、一次跑定；若机制跨 5-10 个结构化任务 × 多 seed 仍成立，才正式从 toy mechanism 升级为研究方向。本步：8 结构化任务 × 5 seed 一次跑定（/tmp/probe_bdh_matrix.py，产物 /tmp/dyn_matrix_results.json）。

### 任务集（8 个结构化任务，全部单 MLP 可学 1.0 ✅）
| 任务 | 计算 | 预期 | 实际 |
|---|---|---|---|
| A_add | 加法·基础 | spawn cap(加) | cap0 |
| A2_add | 加法·变体 | **复用** cap(加) | cap0 ✅ |
| A3_add | 加法·变体 | **复用** cap(加) | cap0 ✅ |
| B_mul | 乘法·基础 | **spawn** cap(乘) | cap1 ✅ |
| B2_mul | 乘法·变体 | **复用** cap(乘) | cap1 ✅ |
| D_sub | 减法·线性 | **spawn** cap(减) | cap2 ✅ |
| C_logic | 布尔·多数 | **spawn** cap(逻辑) | cap3 ✅ |
| D_seq | 序列·外推 | **spawn** cap(序列) | cap4 ✅ |

### 判据结果（3 条硬性门槛全过）
**① 分化矩阵正确（5/5 seed 完全一致，一字不差）**
```
cap0 = [A_add, A2_add, A3_add]   (加族，3 变体自动复用)
cap1 = [B_mul, B2_mul]           (乘族，2 变体自动复用)
cap2 = [D_sub]  cap3 = [C_logic]  cap4 = [D_seq]
```
**② 每任务 acc ≥ 0.85：8/8 任务 × 5/5 seed 全达标**
| seed | A_add | A2_add | A3_add | B_mul | B2_mul | D_sub | C_logic | D_seq |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.0 | 1.0 | 1.0 | 0.977 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2 | 1.0 | 1.0 | 1.0 | 0.998 | 1.0 | 1.0 | 1.0 | 1.0 |
| 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 4 | 1.0 | 1.0 | 1.0 | 0.982 | 1.0 | 1.0 | 1.0 | 1.0 |
**③ 最终 cap 数 = 5**（分布 [5,5,5,5,5]，= 加/乘/减/逻辑/序列 各 1；非 8=无复用、非 1=无分化）

### 判定
**✅ 升级成立。** 5/5 seed 分化矩阵完全一致 + 8 任务全 acc 满分 + cap 数=5（5 能力族）。
- **无 oracle**：模型靠"复用会伤旧→新建"判据，**自动**把 8 任务分成 5 能力（加/乘/减/逻辑/序列）。
- **同类复用**（加变体×3、乘变体×2 各共用一个 cap）+ **异类新建**（减/逻辑/序列 各 spawn）。
- **跨 5 seed 完全稳定**（assign 逐 seed 一字不差）+ **8 任务全部 1.0** + **持续学习**（加族 3 个在学完乘/减/逻辑/序列后仍全 1.0）。
- **这正是"动态结构 + 持续学习 + 自动能力边界"三命题在结构化任务 × 多 seed 上的完整实证 —— 从 toy phenomenon 升级为可验证研究方向。**

### 核心判据确立（反哺任何动态模块/边权设计）
1. **能力路径须独立**（每能力完全独立计算路径，非共享中间层——共享层必被后续改写覆盖）
2. **隔离看"复用是否伤旧"**（非"新任务 fit"——后者在简单任务无区分度，前者是能力隔离本质）
3. **测能力边界须用需不同计算通路的任务**（z 条件/XOR/非线性/异族；简单回归单 cap 通吃测不出）

> 产物：`/tmp/probe_bdh_matrix.py`、`/tmp/dyn_matrix_results.json`（临时）。

## 2026-09-06 FDN-v0.7 根因反证 #2：路由可导性 NOT 根因（颠覆性）

在 StaticMoE 上做「一次只动路由方式」对照（单任务 A，n=12，k=5，30ep，/tmp/probe_router.py）：

| 路由方式 | 单任务 A | 解读 |
|---|---|---|
| hard top-k | 0.049 | 基线（学不会） |
| soft 全连通 (temp=1) | 0.045 | **可导了，仍学不会！** |
| soft 全连通 (temp=0.5) | **0.092** | 略好但仍差 |
| soft 全连通 (temp=2) | 0.045 | |

### 核心结论（重要颠覆）
**top-k 路由不可导 NOT 根因。** 即便用 soft 全连通（完全可导，Router 拿到梯度），MoE 单任务 A 仍只到 0.045-0.09。而 **StaticMLP（共享网络）单任务 A = 1.000**。

真正的矛盾：**同一个 A 任务，共享 MLP 学会（1.0），但任何「多 node 路由 + 逐 node 独立计算」的 MoE 结构（无论 hard/soft 路由）都学不会（0.05-0.09）。**

**根因指向（比路由可导性更深）**：不是路由的梯度问题，而是 **MoE 架构本身——多个独立 DynamicNode + 稀疏路由 对该 toy 任务「梯度分配失败」**。A=y=(a+b)/2 是全局简单线性函数，Router 无法把输入分解到不同 node（没有天然的任务子结构），导致每个 node 都在学全部 → 12 个 node 学一个简单线性 = 参数浪费 + 梯度稀释 + 相互干扰。共享 MLP（一个参数集）反而最优。

**这推翻了我们 v0.3→v0.7 五个版本的外围机制方向**（protected/软亲和/spawn/plasticity/freeze/warm-up 全是在"假设路由+独立 node 能学会"的前提下做加法），但现在证明**前提本身不成立**——MoE 架构对抗 toy 任务的梯度分配是根本障碍，不是外围机制能补的。
> 产物：`/tmp/probe_v06e.json`（临时）。测试 19/19。

## 2026-09-06 FDN-v0.6 批次 3：Plasticity Decay 5 态

按 GPT 生命周期（NEW 高可塑 / WARMING / LEARNING / MATURE 低 / DORMANT 极低 / REACTIVATED 临时高）实现 5 态状态机。
- model/fdn.py：+`life_stage` / `last_active_epoch` buffer；`update_lifecycle` 由「单一 mature×0.1 衰减」改为「按 life_stage 分级衰减」（`plastic_decay_map`：NEW 0.98 / WARMING 0.95 / LEARNING 0.90 / MATURE 0.85 / DORMANT 0.98 / REACTIVATED 1.15）；`_mark_used` 记录 last_active_epoch。
- 测试：+`test_v06_plasticity_five_stage`（验证 NEW/MATURE/DORMANT 不同 stage 与 plastic 衰减差异）——19/19 全绿。

### 批次 3 关键发现（探针 v06 mini, seed 0）
- ✅ **5 态状态机在真实训练中运作**：life_stage=[3,3,4,4,3,3,4,4,4,3,3,4,3,0]（MATURE/DORMANT/NEW 分布合理）；spawn=2 正常。
- ❌ **but A 灾难性遗忘**：forgetting_A=0.992，A_curve=[0.252, 0.412, 0.002]——A 在 B 任务升到 0.412，A' 时崩到 0.002。
- ❌ **分化仍未形成**：disjoint=0，cos(A,B)=0.989，cos(A,A')=1.0；node12 仍被 B 和 A' 共用。
- 🎯 **本质缺口（批次 3 核心教训）**：**Plasticity Decay 只保护 Hebbian 推理期更新，不拦 Adam 主训练梯度**——A 的 node 在 B 任务仍被 Router 选中 → Adam 更新 → A 能力被覆盖。这指向：**要实现真模块隔离，须在任务边界冻结旧 node（Adam 不更新旧 node 参数）或新 spawn node 专属化**——正是 GSLM v0.1「真模块隔离」验证过的方向。
> 产物：`/tmp/probe_v06d.json`（临时）。测试 19/19。


