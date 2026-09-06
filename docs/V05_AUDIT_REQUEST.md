# FDN-v0.5 外部 GPT 二审材料包（AUDIT REQUEST v0.5）

> **审计对象**：`github.com/sinoclaw/fdn`，commit **`2e28dc1`**（FDN-v0.5：多 seed 复现 + 等难度任务 + Node Specialization Matrix）
> **一审裁定**（docs/GPT_AUDIT_V04.md）：先多 seed 验证 w=0.5 → 换成等难度任务 → Node Specialization Matrix。**本材料为一审裁定的执行结果，请二审裁定下一步方向。**
> **三层对账**：① commit `2e28dc1` + ② 原始 JSON（results/summary_v05_seed{0-5}.json、summary_v05_equi.json）+ ③ 报告（results/V05_MULTISEED.md / V05_REPORT.md）

---

## 1. 背景（FDN 假说 + 上轮裁定）

FDN-v0：「全动态模块化路由网络」——每 Node 独立计算路径（in_proj/循环/out_head/key），Router 用 q=W_q·x 与 Node key 点积 top-k 路由，Node 可 spawn/merge 生长。
**核心假说**：学会 A 的 Node 经历 B、C 后被 A'（同分布）重新调用仍会做 A（长出并调用）。

**上轮（v0.4-lite w=0.5）审计**：GPT 认可「A 保住 + B 学会 + A' 调用」是漂亮信号，但**不同意叫"动态模块形成机制"**——因为 B/C 未形成独立模块（C_logic=0.002）。GPT 裁定下一步：**多 seed 验证 w=0.5（头号不确定性）+ C 换等难度任务 + Node Specialization Matrix 指标**，复现成功再进入 Dynamic Neural Ecology。

---

## 2. 批次 1：多 seed 复现（w=0.5, seed 0..5, 40ep, core）

| seed | A_end | forgetting | B_mul | C_logic | A2 | overlap | nodes |
|---|---|---|---|---|---|---|---|
| 0 | 0.018 | **0.904** | 0.035 | 0.086 | 0.031 | 0.538 | 15 |
| 1 | 0.084 | 0.0 | 0.035 | 0.045 | 0.021 | 0.429 | 15 |
| 2 | 0.188 | 0.0 | 0.0 | 0.0 | 0.742 | 0.700 | 14 |
| 3 | **0.973** | 0.0 | 0.002 | 0.053 | 1.0 | 0.800 | 13 |
| 4 | 0.223 | 0.0 | 0.084 | 0.070 | 0.254 | 0.875 | 14 |
| 5 | 0.061 | 0.613 | 0.006 | 0.010 | 0.502 | 0.636 | 14 |

**结论**：
- ❌ **w=0.5 不跨 seed 复现**：A_end 0.018→0.973（>50 倍），forgetting 0→0.904，**仅 seed3 达标**（A>0.4/忘≈0/A2>0.35/B>0.15）。**GPT「最大不确定性」警告被证实。**
- ❌ **B 全部 <0.1**（0/6 达 0.15 阈值）。
- ✅ overlap 4/6 >0.5（A/A' 复用机制在，但精度不稳）。
- 🎯 **所有 seed `task_affinity_disjoint_frac=0.0`**（无模块分化）。
- 🎯 **seed3 的 A=0.973 是"假象"**：A_curve=[0.08,0.041,0.064,0.973]，A 一直没学会，A' 时才虚高（恰命中大输出 node）。

---

## 3. 批次 2：等难度任务（D_sub 替代 C_logic，equi 序列）

`A→B→D_sub→A2_add`（seed 0, 40ep）：final_acc = A=0.152, **B=0.0**, D_sub=0.139, A2=0.184；forgetting_A=0；A_curve=[0.01,0.115,0.127,0.152]。

- ✅ **D_sub=0.139 >> C_logic=0.002** → **GPT「C 难是布尔任务本身难，非机制缺陷」建议被证实**（等难度任务能学到）。
- ✅ **forgetting_A=0**（A 保留，A_curve 单调升）。
- ❌ **B_mul=0.0**（B 完全没学会）、disjoint=0。

---

## 4. 批次 3：Node Specialization Matrix（核心证据）

specialization_cos（行余弦，**越高越相似=越不分**）：

| 任务对 | cos | 应该 | 实际 |
|---|---|---|---|
| A~A2 | 0.984 | 高（A/A' 复用 ✓）| 0.984 ✓ |
| A~B | 0.927 | **低（应分化）** | **高 ❌** |
| A~D | 0.952 | **低** | **高 ❌** |
| B~D | 0.954 | **低** | **高 ❌** |
| B~A2 | 0.884 | 低 | 高 ❌ |

specialization_matrix 显示 A/B/D/A' 命中**几乎完全相同的 node 分布**（集中在 node 5,7,8,9,14）——**4 个任务共用同一批 Node，无任何分化**。

---

## 5. 总判定（对照一审判据）

| GPT 判据 | 达标? | 证据 |
|---|---|---|
| 多 seed 复现（A>0.4/忘≈0/A2>0.35/B>0.15，5/6） | ❌ | 仅 seed3，A_end 跨度 50 倍，B 全 <0.1 |
| 等难度任务（A/B/D 均>0.3） | ❌ | D_sub=0.139 有进步但 <0.3，B=0.0 |
| 模块分化（A-A2 高 / A-B/D 低） | ❌ | cos 全 >0.88，disjoint=0，矩阵全同 |

**核心结论**：
- ✅ **GPT 两条具体建议被验证**：① 换等难度任务有效（D 0.139 > C 0.002）；② 不确定能否复现的警告被证实（w=0.5 不跨 seed 稳定）。
- ❌ **「模块分化机制」未实现**——这是 FDN 当前最根本缺口。所有任务 Node 高度重叠 → 不分摊 → B 学不好、A 易被覆盖/假象、seed 不稳定。
- **意义**：v0.3→v0.4-lite 解决了表面（A 遗忘、w 强度），但底层「**任务如何分化到不同 Node**」尚未解决——正是 GPT 指向 Dynamic Neural Ecology 的核心（Node 自主 specialization）。

---

## 6. 请二审裁定的问题

1. **是否认同「根因 = 任务间无模块分化（disjoint=0 / 行余弦全高）」**？还是可能另有归因（如 Router 未学到任务区分、top-k 门控本身脆弱）？

2. **模块分化缺失，最可能是下面哪个缺口**（请裁定优先级）：
   - a) **Router 学不到任务区分**：soft task bias 只加在 query 偏移上，但 top-k 路由仍倾向共选高亲和 node（无 novelty/spawn）。
   - b) **无 novelty-driven spawn**：新任务未触发专属 node 生长，全部挤进初始共享 node。
   - c) **缺少 module specialization 正向压力**：没有机制奖励"不同任务用不同 node"（hard 隔离已证伪，需软的、学习到的）。

3. **三个候选下一步（请选/排序）**：
   - ① **Novelty Detector + 专属 spawn**：新任务超 novelty 阈值时强制 spawn 专属 node + 局部训练（避免挤压共享 node）——这是否会重蹈 v0.4 硬隔离覆辙？
   - ② **强 task-conditioned routing**：让 task_emb 直接参与 top-k 选择（而非仅 query 偏移），使不同任务偏好不同 node。
   - ③ **Dynamic Neural Ecology 第一步**：Node 自主 specialization（观察输入/收益/使用率 → 决定可塑/复制/休眠），从 routing 转向 ecology。

4. **是否认为当前证据足以判断"FDN 尚未达到动态模块形成机制"**，还是需先本地诊断 Router 为何不学任务区分？

---

## 7. 三层对账清单

| 版本 | commit | 原始 JSON | 报告 |
|---|---|---|---|
| v0.5 批次1 | 2e28dc1 | results/summary_v05_seed{0-5}.json | results/V05_MULTISEED.md |
| v0.5 批次2 | 2e28dc1 | results/summary_v05_equi.json | results/V05_REPORT.md |
| v0.5 批次3 | 2e28dc1 | (同上内嵌 specialization) | results/V05_REPORT.md |

---
*（材料由团队按 commit + 原始 JSON + 报告三层对账整理，供外部独立审计裁量下一步方向。）*
