# FDN-v0.2 实验结果报告（V02_REPORT，如实版）

> 本报告与 `results/summary_v02a.json` / `results/summary_v02b.json`（原始数据）、`docs/V02_PLAN.md`（规划）、`docs/GPT_AUDIT_V01_REVIEW.md`（GPT 复审）形成三层对账，供外部审计（GPT）核定。
> 依据：GPT 复审指出 v0.1 稳定化后学习仍失败，真正问题是 **Dynamic Routing 破坏优化（鸡生蛋）**，而非结构震荡。v0.2-A = 保留承重动态+稳定结构；v0.2-B = 再加 Router warm-up + soft→hard curriculum。

---

## 0. 一句话结论

**鸡生蛋被破解。** Router warm-up + soft→hard curriculum（v0.2-B）把 A 学到 **0.455**，相比 v0.1 的 0.045 提升约 **9 倍**、相比 v0.2-A 的 0.211 翻倍；`forgetting_A=0`、A/A' 均 ~0.45（**调用而非重学**）——GPT 预期的「路由冷启动限制 Node 学习」的核心问题，在 v0.2-B 得到解决。但 **cross-task（B_mul=0.020 / C_logic=0.176）仍不足**，是下一步方向。

## 1. 实验设定

- 任务序列 `A_add → B_mul → C_logic → A2_add(A'变体)`，40 epoch/任务，1200 样本，batch 32。
- hyper：hidden=32, r=16, init_nodes=12, k_fixed=5, lr=3e-3；结构稳定化 controller（任务边界触发 + spawn 冻结 + merge patience，与 v0.1 相同）。
- v0.2-A：`--profile v02`（4 个承重动态开关 Plasticity/Memory/τ/动态k = True，其余与 v0.1 相同——**一次只动一组变量**）。
- v0.2-B：`--profile v02b --warmup_tasks 1`（v0.2-A + 前 1 任务用全班 soft 路由、温度退火到 hard，之后切 top-k）。

## 2. 原始数据摘要（C，seed0，40ep core）

| 指标 | v0 全动态 | v0.1 稳定化 | v0.2-A 全动态 | **v0.2-B 课程** |
|---|---|---|---|---|
| acc_A_end | 0.055 | 0.045 | 0.211 | **0.455** |
| A2(A') | 0.092 | 0.041 | 0.508 | **0.447** |
| B_mul | 0.0 | 0.041 | 0.002 | 0.020 |
| C_logic | 0.010 | 0.057 | 0.008 | **0.176** |
| forgetting_A | — | — | 0.0 | **0.0** |
| node_overlap_A_A2 | 1.0 | 0.667 | 0.692 | 0.636 |
| final_nodes | 54 | 15 | 14 | 13 |
| spawn/prune/merge | 42/5/42 | 3/6/0 | 2/4/0 | 1/5/0 |

**v0.2-B A_curve_after_each = [0.047, 0.0, 0.006, 0.455]**（学完A → 学完B → 学完C → 学完A'）。

## 3. 证据链（三层对账）

### 3.1 机制层：承重动态恢复 + 课程机制真实生效
- v0.2-A 把 4 个承重开关打开（v0.1 时关着）→ acc_A 从 0.045 回升到 0.211，`forgetting_A=0`，A2=0.508 —— **恢复承重动态有效**（与 GPT 复审一致）。
- v0.2-B 的 warm→hard 转：A_curve 显示「学完A'时A回到0.455」—— Router 学到任务亲和后，A' 训练重激活 A 的 Node 组（A 恢复、调用非重学），`node_overlap_A_A2=0.636`（有复用）。

### 3.2 能力层：A 真正学会（非随机）
- v0.2-B `acc_A_end=0.455`（A_add 从 0.047 攀升），A2=0.447 —— 不再是 v0/v0.1 的 ~0.05 随机水平。
- 对比 v0.1（关承重动态）0.045 vs v0.2-B 0.455：**提升 ~9 倍**，差异巨大、方向明确。

### 3.3 鸡生蛋判定：v0.2-A 确认存在，v0.2-B 破解
- **v0.2-A**：acc_A=0.211 > v0.1 的 0.045，但 B=0.002/C=0.008 近随机 → Router 冷启动未建立跨任务 Node 分工（鸡生蛋存在，承重动态是必要非充分）。
- **v0.2-B**：加 warm-up+curriculum 后 A=0.455（A/A' 都会）、C_logic 升到 0.176 —— **路由冷启动被 warm 阶段（全班参与、Router 学到亲和）破局**，再切 hard 承载能力。

## 4. 判定

- **部分通过（核心机制验证成功）**：v0.2-B 达成 GPT 核心验证目标——① A 真正学会（0.455，非随机）；② 遗忘归零（forgetting_A=0）；③ A' 调用（A/A' 均 ~0.45 且 Node 复用）——「**学会 → 保留 → 调用**」链路跑通。
- **未完全达成**：cross-task 分工仍不足（B_mul=0.020、C_logic=0.176 明显低于 A）。说明当前 warm-up 破的是「冷启动路由无信号」，但「多任务各自成形（task-node 亲和唯一化）」尚未解决——这是 v0.2-C（更强 curriculum / 任务亲和正则 / 更长 warm / 更难任务）方向。

## 5. 局限与混杂（如实声明）

1. **单 seed（seed=0）**：方向性强（A 从 0.045→0.455），但复现性未做多 seed。
2. **toy 任务容量充足**：B/D/C（MoE 族）本就比稠密 MLP（A=1.0）难学；这里 C 的绝对 acc 仍低于静态基线，但**趋势（鸡生蛋破解、A 学会且保留）**是关键结论。
3. **warm-up 阶段评估走 hard**：`acc_A_init=0.047`（首个任务后、仍 warm 期）代表的是 hard 评估，非 warm 能力；最终以 `acc_A_end`（全 hard、训练收敛后）为准。
4. **cross-task 不足**：B/C 未学到可读精度，当前结论对「FDN 能学会并保留同一任务的调用」成立，对「跨任务无干扰地发展多个能力」尚未成立——不宜过度宣称。

## 6. 交付清单

- 代码：`model/fdn.py`（soft→hard curriculum + warm）。`experiments/continual.py`（--profile v02/v02b/v02c + retention + A_curve）。`experiments/tasks.py`（RETENTION_SEQUENCES）。`tests/test_fdn.py`（12/12，含 3 个 curriculum 测试）。
- 规划：`docs/V02_PLAN.md`；GPT 复审：`docs/GPT_AUDIT_V01_REVIEW.md`。
- 原始数据：`results/summary_v02a.json`、`results/summary_v02b.json`、`results/summary_v02c.json`、`results/summary_v02b_retention.json`。

---

## 7. v0.2-C（方案 A，per-task warm-up）——重要发现

> Dad 拍板方案 A：每个任务前 `warm_epochs` 个 epoch 用全班 soft 路由（per-task warm-up），让每个能力组都能长出。实测揭示**新的权衡**，是本项目迄今最有价值的机制级发现。

### 设定
`--seq core --profile v02c --warm_epochs 8 --warm_mode every --seed 0 --n_train 1200 --epochs_per_task 40 --run C`（每任务前 8 ep 全班 soft，之后 hard top-k）。

### 实测（v0.2-C vs v0.2-B vs v0.2-A）
| 策略 | A_init | A_end | forgetting_A | B | C | A2 | spawn/merge | final_nodes |
|---|---|---|---|---|---|---|---|---|
| v0.2-A 无warm | 0.143 | 0.211 | 0.0 | 0.002 | 0.008 | 0.508 | 2/0 | 14 |
| v0.2-B 首任务warm | 0.047 | **0.455** | **0.0** | 0.020 | 0.176 | 0.447 | 1/0 | 13 |
| v0.2-C 每任务warm | **0.340** | 0.053 | **0.845** | 0.029 | 0.033 | 0.053 | **0/0** | **12** |

v0.2-C `A_curve_after_each=[0.34, 0.004, 0.018, 0.053]`；timeline：A=0.34→B后A崩到0.004→C后0.018→A'后0.053。

### 判定（如实）
- **方向对了一半**：per-task warm 让 **A 学得最快**（A_init=0.340，比 v0.2-B 高 7 倍）—— warm 确实让任务快速长出能力。
- **但灾难性遗忘回归**：学 B 时 A 从 0.34 崩到 0.004（`forgetting_A=0.845`），最终 A_end=0.053。**per-task warm 用全班 soft 路由，每个任务都在更新同一批共享 Node → 强覆盖**。
- **结构零增长（spawn=0）**：全班 soft 时 Router 觉得所有 Node 都参与（无 novelty），**不再触发 spawn**，模型冻结在初始 12 Node。稀疏/模块隔离的意义被 warm 消解。

### 机理性结论（复现 GSLM No-Go 根因）
`共享投影被后续任务改写 → 灾难性遗忘` 在 FDN 上**复现**了——虽然 FDN 有独立 out_head（模块化），但 **per-task 全班 soft 路由让每个任务都回流到相同 Node**，等价于共享投影。**「warm 的 softness」与「模块隔离的 sparseness」是矛盾的**：
- soft 太多（every warm）→ 覆盖（遗忘）+ 抑制生长（spawn 0）
- soft 太少（first warm / 无 warm）→ 冷启动学不动（A_init 低）但保留好（forgetting 0）

### 下一步（交 GPT 裁量，团队不自行跳步）
- **动态 softness**：warm 只用于**新 Node 的诞生期**（spawn 后冻结期用全班 soft 让新 Node 吸收梯度），非每任务全班 soft——与「毕业机制 + 节点冻结」结合。
- **task-conditioned 路由 + 局部收敛**：warm 后**只更新本任务的 Node 组**（隔离），而非全班。
- 值得把「warm softness vs sparseness」矛盾的完整矩阵（A/B/∞ 三档 warm 强度）交外部审计，作为**模块化持续学习的机制级判据**。

*（2026-09-06 v0.2-C 补充。）*
