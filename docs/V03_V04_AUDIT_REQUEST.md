# FDN-v0.3 → v0.4-lite 外部 GPT 审计材料包（AUDIT REQUEST）

> **审计对象**：`github.com/sinoclaw/fdn`（FDN-v0，全动态模块化路由网络）
> **审计范围**：三个提交 `32066e9`（v0.3）、`cea59e2`（v0.4 硬隔离，反证）、`c3d8947`（v0.4-lite 软任务亲和，FND 迄今最优）
> **三层对账**：每个版本均有①代码 commit + ②原始 JSON（results/summary_*.json）+ ③报告（results/V03_REPORT.md / V04_REPORT.md / V04LITE_REPORT.md）
> **请外部 GPT 裁定**（见末尾 5 个问题）

---

## 1. 问题背景（FDN 假说）

FDN-v0 是一个「全动态模块化路由网络」：每个 Node 是**独立计算路径**（独立 in_proj/循环/out_head/key），Router 用查询向量 q=W_q·x 与每 Node key 的点积做 top-k 路由，Node 可 spawn/merge 生长。

**核心假说（A→B→C→A' 调用链）**：学会任务 A 的 Node，经历 B、C 后被 A'（同分布任务）重新调用，仍会做 A——即「长出并调用」，而非 MoE 变体。

**研究路线**（GPT 上轮已确认方向）：
```
v0 结构乱 → v0.1 结构稳定 → v0.2 发现 Router cold-start → v0.2B 证明 first-task warm-up 可行
→ v0.2C 发现 global soft routing 造成能力覆盖 → v0.3 新模块soft旧模块protected → v0.4 硬隔离(反证) → v0.4-lite 软任务亲和(最优)
```

---

## 2. 全历史 C 模型矩阵（seed 0, 40ep, A→B→C→A' core 序列）

| 版本 | 机制 | A_init | A_end | forget | overlap_A_A2 | B_mul | C_logic | A2 | node |
|---|---|---|---|---|---|---|---|---|---|
| v0 | 全动态 | 0.061 | 0.055 | 0.097 | 1.0 | 0.0 | 0.010 | 0.092 | 54 |
| v0.1 | 稳定化 | 0.016 | 0.045 | 0.0 | 0.667 | 0.041 | 0.057 | 0.041 | 15 |
| v0.2-A | 稳定+全动态 | 0.143 | 0.211 | 0.0 | 0.692 | 0.002 | 0.008 | 0.508 | 14 |
| v0.2-B | 首任务warm | 0.047 | 0.455 | 0.0 | 0.636 | 0.020 | 0.176 | 0.447 | 13 |
| v0.2-C | 每任务warm | 0.340 | 0.053 | **0.845** | 0.727 | 0.029 | 0.033 | 0.053 | 12 |
| v0.3 | 保护式 | 0.006 | 0.285 | 0.0 | 0.667 | 0.051 | 0.041 | 0.391 | 13 |
| v0.4 | 硬隔离 | 0.047 | 0.023 | 0.5 | **0.0** | 0.031 | 0.004 | 0.090 | 37 |
| **v0.4-lite w=0.3** | 软偏 | 0.115 | 0.104 | 0.102 | 0.889 | 0.152 | 0.029 | 0.076 | 12 |
| **v0.4-lite w=0.5** | 软偏 | 0.236 | **0.545** | **0.0** | 0.700 | **0.209** | 0.002 | **0.477** | 12 |
| v0.4-lite w=0.8 | 软偏 | 0.078 | 0.020 | 0.75 | 0.700 | 0.0 | 0.109 | 1.0 | 12 |

---
## 3. 三个待审提交的关键结论

### 提交 ① `32066e9` — FDN-v0.3 Protected Expert Formation
**机制**：per-node 生命周期（maturity/competence/node_epoch）+ ① A 新 Node 专属 warm-up（未成熟 Node 门控放大吸梯度，成熟 Node hard protected）+ ② Competence Lock（loss↓/usage↑/entropy↓→maturity 升，成熟后 plasticity 衰减）+ ③ Re-activation（reactivate_score 复用成熟 A-Node）。

**结果**：A 遗忘从 v0.2-C 的 **0.845 → 0**（旧模块被保护、不被覆盖），A_end=0.285，A2=0.391（A' 调用恢复）。**方向正确**。但 B_mul=0.051、C_logic=0.041 仍低，且 reactivation 显示 A/B/C 命中成熟 Node 重叠大（全含 11/6）→ **B/C 未形成独立 Node 组，跨任务能力不足**。

### 提交 ② `cea59e2` — FDN-v0.4 任务亲和约束（硬隔离，**反证**）
**机制**：task_owner 归属 + _task_mask **硬屏蔽**其他任务 Node；新任务 spawn 专属 Node 组（init_per_task）。

**结果**：**反证**。node_overlap_A_A2=**0.0**（A 与 A'=task3 的 Node 完全不相交），A_end=0.023、forgetting=0.5、prune 29 次。
**判定**：硬隔离（强制不相交）**违反 FDN 核心假说**——它切断了「A' 调用 A 的 Node」这一跨任务复用（v0 已证 A/A' IoU=1.0）。**任务亲和约束应为软偏好，硬隔离是错误路径。**

### 提交 ③ `c3d8947` — FDN-v0.4-lite 软任务亲和（**迄今最优**）
**机制**：`q = W_q·x + w·task_emb[task_id]`（可学习 task_emb 加 query 偏移），**不硬屏蔽**（task_constraint=False），保留 v0.3 全部机制。扫描 w∈{0.3,0.5,0.8}。

**结果**：**w=0.5 为 FDN 迄今最优**——A=**0.545**（超 v0.2-B 0.455 纪录）+ forgetting=**0** + B=**0.209**（v0.2-B 的 10 倍）+ A2=0.477（A' 调用）+ overlap=0.700（A/A' 复用完整）。
- w=0.3 太弱：A 被 B 挤占（A=0.104），软偏不足以分流。
- w=0.8 太强：task_emb 主导路由 → 破坏 A/A' 复用（A=0.020，forgetting=0.75）。
- **软任务亲和存在最优强度 w≈0.5**——同时保住跨任务复用 + 让难任务（B）学得更多。

---

## 4. 三层对账清单（供核查）

| 版本 | commit | 原始 JSON | 报告 | 相关代码 |
|---|---|---|---|---|
| v0 | 5acc875 | results/summary_v0.json | results/V0_REPORT.md | model/fdn.py |
| v0.1 | cbafad0 | results/summary_v01.json | results/V0_REPORT.md | model/fdn.py |
| v0.2-A/B/C | 8a2b24a | summary_v02a/b/c.json | results/V02_REPORT.md | experiments/continual.py |
| v0.3 | 32066e9 | results/summary_v03.json | results/V03_REPORT.md | model/fdn.py |
| v0.4 | cea59e2 | results/summary_v04.json | results/V04_REPORT.md | model/fdn.py |
| v0.4-lite | c3d8947 | summary_v04lite{,_w05,_w08}.json | results/V04LITE_REPORT.md | model/fdn.py |

---

## 5. 请外部 GPT 裁定的问题

1. **v0.4-lite w=0.5 是否已达 GPT 上轮判据**「A→B→C→A' 且 A 基本不掉、B/C 形成独立 Node」？还是 B/C 仍未真正独立（reactivation 命中 Node 仍略重叠）？

2. **软任务偏的最优强度 w=0.5 是否可信**，还是单 seed（seed=0）+ toy 任务的偶然？需不需要多 seed 复现 + 增加任务难度？

3. **C_logic 偏低（w=0.5 时 0.002，v0.2-C 时 0.176）**：布尔逻辑任务难学，是否应给难任务（C）单独 spa，或在 soft 偏好下叠加「任务专属 Node 只对未成熟任务 spawn」的折中？

4. **下一步方向**：a) 细扫 w={0.4,0.6}；b) 多 seed 复现 w=0.5；c) 增加任务难度/序列长度让容量成瓶颈；d) 交「Dynamic Neural Ecology」真正落地（Node age/state/specialization/usage/memory/competence 完整生命周期）；e) 其他？

5. **是否认为 FDN-v0.4-lite w=0.5 已经接近「动态模块形成机制」**（非 MoE 变体）？若是，具象化缺失的一环是什么？

---
*（审计材料由团队按 commit + 原始 JSON + 报告三层对账整理，供外部独立审计裁量。）*
