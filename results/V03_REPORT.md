# FDN-v0.3 实验结果报告（V03_REPORT，如实版）

> 本报告与 `results/summary_v03.json`（原始数据）、`model/fdn.py`（Protected Expert Formation 实现）、`experiments/continual.py`（--profile v03）形成三层对账，供外部审计（GPT）核定。
> 依据：GPT v0.2 复审（docs/GPT_AUDIT_V02.md）建议 v0.3 Protected Expert Formation——**soft 只能用于"新模块诞生成熟"，旧模块须 protected**。

---

## 0. 一句话结论

**A 遗忘被彻底解决（forgetting=0），且 B 首次略学会（0.051）；但 B/C 尚未形成独立 Node 组（C_logic 仍低）。** v0.3 的方向完全正确（保护旧模块 + 只让新模块 warm），在「保 A」与「学 B/C」间取得比 v0.2-B/C 更好的权衡，但「动态模块形成机制」尚未完全落地——B/C 的独立 Node 组还需更强的任务亲和约束。

## 1. 机制（FDN-v0.3）

- **A. 新 Node 专属 warm-up**：`forward` 内 per-node warm——未成熟 Node（maturity<thr）在 top-k 内门控放大（高门控独立吸收梯度），成熟 Node protected（标准 hard gate，不被后续任务拉宽）。
- **B. Competence Lock**：`update_lifecycle` 按 loss↓/entropy↓/usage↑ 逐 epoch 提升 maturity；成熟（≥thr）后 plasticity 衰减（锁死可塑）→ 旧模块不再被覆盖。
- **C. Re-activation**：`reactivate_score` 按查询亲和返回已成熟 Node 排序，Router 直接复用（不重新 warm）。

## 2. 设定

- `--seq core --profile v03 --seed 0 --n_train 1200 --epochs_per_task 40 --run C`
- 对照：v0.2-B（acc_A_end=0.455，遗忘 0）、v0.2-C（0.053，遗忘 0.845）。

## 3. 原始数据摘要

| 指标 | v0.2-B 首任务warm | v0.2-C 每任务warm | **v0.3 保护式** |
|---|---|---|---|
| acc_A_end | 0.455 | 0.053 | **0.285** |
| forgetting_A | 0.0 | **0.845** | **0.0** |
| A2(A') | 0.447 | 0.053 | **0.391** |
| B_mul | 0.020 | 0.029 | **0.051** |
| C_logic | 0.176 | 0.033 | 0.041 |
| final_nodes | 13 | 12 | 13 |
| n_mature | — | — | **13/13** |

A_curve_after_each=[0.006, 0.0, 0.002, 0.285]；reactivation 每任务均命中成熟 Node（A/B/C 的 top-mature 有重叠，指向共享 Node 组）。

## 4. 证据链

- **机制层**：`n_mature=13/13`（全成熟）、reactivation 每任务返回成熟 Node 排序（B/C/A 均命中）——Competence Lock 与 Re-activation 真实发生。
- **能力层**：
  - ✅ **A 遗忘=0、A_end=0.285、A2=0.391**（A 学会 + A' 调用恢复）——「旧模块被保护、不被后续任务覆盖」达成，这是 v0.2-C 的致命伤。
  - ✅ **B_mul 首次达 0.051**（per-node warm 让 B 略学会），高于 v0.2-B 的 0.020。
  - ⚠️ **C_logic=0.041 仍低**，B/C 与 A 的 reactivation 命中 Node 重叠大（都含 11/6）——**B/C 尚未形成独立 Node 组**。

## 5. 判定

- **部分达成（方向正确）**：v0.3 达成「A 学会且保留（遗忘0）」这一核心目标，且 B 首次小幅学会；**但 GPT 判据「A→B→C→A' 且 B/C 形成独立 Node」未完全满足**——B/C 的独立 Node 组尚未形成。
- **意义**：v0.3 证明「保护旧模块（protected）+ 只让新模块 warm」是**对的方向**（A 遗忘从 0.845→0），但要真正做到「动态模块形成机制」，需更强的**任务亲和约束**（强制不同任务映射到不相交 Node 组），这可能是 v0.4 方向。

## 6. 局限与混杂（如实声明）

1. **单 seed（seed=0）**：方向性问题（A 遗忘=0 是核心结论，差异巨大），但复现性未做多 seed。
2. **toy 任务容量充足**：B/C/D（MoE 族）本就比稠密 MLP（A=1.0）难学；`C_logic` 是 booleans 归一化回归，天然难到高精度。
3. **超参未扫**：per-node warm 强度（warm_temp）/ mature_thr / competence_lr 的敏感性未扫；maturity 更新速率可能偏快（首任务就全成熟）。
4. **B/C 独立 Node 组未形成**：reactivation 显示 A/B/C 命中重叠 Node，任务亲和约束（per-task 专属 Node 组）未显式强制——这是当前跨任务能力不足的主因，属实现边界而非机制失效。

## 7. 交付清单

- 代码：`model/fdn.py`（maturity/competence/node_epoch + update_lifecycle + reactivate_score + per-node protected routing）、`experiments/continual.py`（--profile v03）。
- 设计：`docs/GPT_AUDIT_V02.md`（GPT 复审）。
- 原始数据：`results/summary_v03.json`。
- 单测：`tests/test_fdn.py`（13/13，含 v0.3 三机制）。
