# FDN-v0.5 实验结果报告（V05_REPORT，如实版）

> 依据 GPT 裁定（docs/GPT_AUDIT_V04.md）：先多 seed 复现 w=0.5 → 换成等难度任务 → Node Specialization Matrix 指标。本报告汇总三个批次，**核心结论：w=0.5 不跨 seed 稳定，且 FDN 当前架构下任务间无真模块分化（disjoint=0 / 行余弦全高）——模块分化未发生是 B 学不好 + A 不稳定的共同根源。**

---

## 0. 一句话结论

**① w=0.5 的优秀是单 seed 偶然（6 seed 仅 1 达标）；② 等难度任务 D_sub 确实优于 C_logic（0.139 vs 0.002，GPT 换任务建议有效）；③ 但最根本的问题被两个批次同时证实——FDN 当前架构下任务间没有模块分化（task_affinity_disjoint_frac 全 = 0，specialization 行余弦全 >0.88），A/B/D 共用同一批 Node。** 不解决模块分化，多 seed 不稳定、任务学不好是必然。

---

## 1. 设定

- 批次 1（多 seed）：`v04lite, w=0.5, 40ep, core(A→B→C→A')`, seed 0..5
- 批次 2（等难度）：`v04lite, w=0.5, 40ep, equi(A→B→D_sub→A')`, seed 0（D_sub 替代 C_logic）
- 批次 3（指标）：Node Specialization Matrix + 行余弦（新增 evaluation/metrics.py）

---

## 2. 批次 1：多 seed 复现（反证）

| seed | A_end | forgetting | B_mul | C | A2 | overlap | nodes |
|---|---|---|---|---|---|---|---|
| 0 | 0.018 | **0.904** | 0.035 | 0.086 | 0.031 | 0.538 | 15 |
| 1 | 0.084 | 0.0 | 0.035 | 0.045 | 0.021 | 0.429 | 15 |
| 2 | 0.188 | 0.0 | 0.0 | 0.0 | 0.742 | 0.700 | 14 |
| 3 | **0.973** | 0.0 | 0.002 | 0.053 | 1.0 | 0.800 | 13 |
| 4 | 0.223 | 0.0 | 0.084 | 0.070 | 0.254 | 0.875 | 14 |
| 5 | 0.061 | 0.613 | 0.006 | 0.010 | 0.502 | 0.636 | 14 |

- **A_end 0.018→0.973（>50 倍）**，forgetting 0→0.904，仅 seed3 达标 → **w=0.5 不跨 seed 复现**。
- B 全部 <0.1（0/6 达 0.15 阈值）。
- overlap 4/6 >0.5（A/A' 复用机制在，但精度不稳）。
- **所有 seed `task_affinity_disjoint_frac=0.0`**（无模块分化）。
- seed3 的 A=0.973 是"假象"：A_curve=[0.08,0.041,0.064,0.973]，A 一直没学会，到 A' 才虚高（恰好命中大输出 node）。

---

## 3. 批次 2：等难度任务（D_sub 替代 C_logic）

`A→B→D_sub→A2_add`（seed 0, 40ep）：

final_acc = A=0.152, B=0.0, D_sub=0.139, A2=0.184；forgetting_A=0；A_curve=[0.01, 0.115, 0.127, 0.152]。

- ✅ **D_sub=0.139 >> C_logic=0.002** → **GPT「C 难不是机制缺陷，是布尔任务本就难」的假设被证实**（等难度任务 D_sub 能学到）。
- ✅ **forgetting_A=0**（A 持续保留，A_curve 单调升到 0.152）。
- ❌ **B_mul=0.0**（B 完全没学会）。
- ❌ disjoint=0。

---

## 4. 批次 3：Node Specialization Matrix（核心证据）

speclialization_cos（行余弦，**越高越差**，低表示分化）：

| 任务对 | cos | 判定 |
|---|---|---|
| A~A2 | 0.984 | 高（本应高：A/A' 复用 ✓） |
| A~B | 0.927 | **高（本应低：应为分化，实际共享）** |
| A~D | 0.952 | **高（本应低）** |
| B~D | 0.954 | **高（本应低）** |

specialization_matrix（行 × node）显示 A/B/D/A' 命中**几乎完全相同的 node 分布**（集中在 node 5,7,8,9,14）——**4 个任务共用同一批 Node，无任何分化**。

---

## 5. 总判定

| GPT 判据 | 达标? | 证据 |
|---|---|---|
| 多 seed 复现（A>0.4/忘≈0/A2>0.35/B>0.15，5/6） | ❌ | 仅 seed3 达标，B 全 <0.1，A_end 跨度 50 倍 |
| 等难度任务（A/B/D 均>0.3） | ❌ | D_sub=0.139（有进步但 <0.3），B=0.0 |
| 模块分化（A-A2 高 / A-B/D 低） | ❌ | cos 全 >0.88，disjoint=0，矩阵全同 |

**核心结论**：
- ✅ **GPT 的两个具体建议被验证**：① 换等难度任务有效（D 比 C 好）；② 不确定性能否复现的警告被证实（w=0.5 不跨 seed）。
- ❌ **但「模块分化机制」仍未实现**——这是 FDN 当前最根本的缺口。所有任务的 Node 高度重叠 → 不分摊 → B 学不好、A 易被覆盖/假象。
- **意义**：v0.3→v0.4-lite 解决了「A 遗忘」和「w 强度」的表层问题，但**底层「任务如何分化到不同 Node」尚未解决**——这正是 GPT 指向 Dynamic Neural Ecology 的核心（Node 自主 specialization）。

---

## 6. 下一步（明确方向）

- **不要继续扫 w / 换任务**，应聚焦**「如何实现模块分化」**：
  1. **诊断为何高相似**：soft task bias 加在 query 上，但 top-k 路由仍倾向于共选高亲和 node → 理论上是 Router 没学到"任务区分"。
  2. **候选方案**（交 GPT 二审后选）：
     - a) 增强 Novelty Detector：新任务超阈值时强制 spawn 专属 node + 局部训练（避免挤压共享 node）。
     - b) 更强 task-conditioned routing：task_emb 影响 top-k 选择（而非仅 query 偏移），使不同任务偏好不同 node。
     - c) Dynamic Neural Ecology 第一步：Node 自主 specialization（观察输入/收益/使用率 → 决定可塑/复制/休眠）。
- **提交 GPT 二审**：把批次 1/2/3 的完整矩阵 + specialization 结论交外部裁量，裁定是否需架构级改造。

---

## 7. 交付清单

- 代码：`evaluation/metrics.py`（+specialization matrix）、`experiments/continual.py`（--seq equi, counts, specialization 输出）、`experiments/tasks.py`（+D_sub）、`tests/test_fdn.py`（+v05 测试，17/17）。
- 原始数据：`results/summary_v05_seed{0..5}.json`、`results/summary_v05_equi.json`。
- 报告：`results/V05_MULTISEED.md`（批次1）、本报告、`results/EXPERIMENT_LOG.md`。
