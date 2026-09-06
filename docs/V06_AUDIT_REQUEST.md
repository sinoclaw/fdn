# FDN-v0.6 外部 GPT 三审材料包（AUDIT REQUEST v0.6）

> **审计对象**：`github.com/sinoclaw/fdn`，commit **`8351d01`**（FDN-v0.6：Node 级 spawn + Plasticity 5 态 + freeze_old）
> **一审**（docs/GPT_AUDIT_V04.md）：v0.4-lite 保留、w=0.5 默认、勿扫 w、下一步多 seed + 等难度 + specialization matrix。
> **二审**（docs/GPT_AUDIT_V05.md）：撤销 w=0.5 稳定、不能称 Dynamic Neural Ecology、下一步 = Node autonomous specialization（v0.6 = Novelty-Spawn + Node Competence + Plasticity Decay）。

---

## 1. 背景（v0.6 意图）

GPT 二审明确：**Robut 负责发现已有能力；Node Ecology 负责产生新能力。** v0.6 = 让 Node 自己产生 specialization（不再靠 Router 的 task_emb oracle）。最简实验 `A→B→A'`（A/B 同难度），判据：`A_end>0.4 / B>0.3 / A2>0.35 / forgetting<0.05 / cos(A,A')>0.7 / cos(A,B)<0.5 / disjoint>0`，**多 seed + specialization matrix + 能力指标三件套**。GPT 明令**不加 oracle / 不调 w / 不接 BDH / 不一次 ABCD**。

---

## 2. 批次实现（三层对账：commit + 代码 + 报告）

| 批次 | 机制 | 状态 |
|---|---|---|
| 1 | Node 级 competence/novelty 状态（`node_novelty=1-cos` / `node_error`）+ 修 competence 信号源 bug | ✅ |
| 2 | Novelty-triggered Spawn：`score=0.4·norm(nov)+0.3·norm(err)+0.3·(1-norm(comp))`，>0.5 触发 | ✅ 触发 |
| 3 | Plasticity Decay 5 态（NEW/WARMING/LEARNING/MATURE/DORMANT/REACTIVATED） | ✅ |
| 3+ | 任务边界 freeze_old（冻结成熟旧 node，`requires_grad=False`） | ✅ 救 A/B |
| 4 | 多 seed 验证（6 seed，40ep，mini A→B→A' + freeze_old） | ❌ 分化未达成 |

---

## 3. 批次 4 多 seed 结果（核心证据）

`A→B→A'`，6 seed，profile v06 + freeze_old，40ep：

| seed | A_end | forgetting | B_mul | A2 | cos(A,A') | **cos(A,B)** | disjoint |
|---|---|---|---|---|---|---|---|
| 0 | 0.021 | 0.847 | 0.0 | 0.020 | 0.999 | **0.997** | 0 |
| 1 | 0.699 | 0.0 | 0.0 | 0.822 | 1.000 | **0.999** | 0 |
| 2 | 0.002 | 0.992 | 0.0 | 0.000 | 1.000 | **1.000** | 0 |
| 3 | 0.016 | 0.857 | 0.713 | 0.008 | 1.000 | **1.000** | 0 |
| 4 | 0.051 | 0.469 | 0.090 | 0.037 | 0.998 | **0.992** | 0 |
| 5 | 0.002 | 0.964 | 0.0 | 0.002 | 0.999 | **0.988** | 0 |

**达标统计**：cos(A,A')>0.7 = 6/6（但全近 1）；**cos(A,B)<0.5 = 0/6**；**disjoint>0 = 0/6**；A_end>0.4 / B>0.3 / forgetting<0.05 各仅 **1/6**。

---

## 4. 关键发现（诚实汇报）

### 4.1 机制都在"启动"但分化未兑现
- spawn 触发（score 0.512，node12 长出，`avg_novelty:0.996 / avg_error:0.945 / sustained:True / task:3`）
- Plasticity 5 态运作（life_stage=[3,3,4,4,...] MATURE/DORMANT/NEW 分布合理）
- 但 **cos(A,B) 全 >0.99、disjoint 全 0——A/B 仍共用同一批 node**。

### 4.2 freeze_old 是承重防遗忘机制，但救不了"没学会"
- 短跑（24ep）"假突破"：forgetting_A 0.992→0.0、A_end 0.002→0.449、B 0→0.904。
- **但完整 40ep 暴露真相：A_curve 起点 0.021（A 从一开始就没学会）**——不是 freeze 失效，是 **A 任务本身学不会（Router 冷启动鸡生蛋）**。

### 4.3 cos(A,A') 全近 1 是"虚高分"
所有任务公用 node，任何两任务行实际近相同，cos 无区分度——不是"A/A' 复用成功"，而是"全都在用同一批 node"。

### 4.4 机制层本质缺口（关键）
**Plasticity Decay 只保护 Hebbian 推理期更新，不拦 Adam 主训练梯度。** A 的 node 在 B 任务仍被 Router 选中 → Adam 照常更新 → A 被覆盖。freeze_old（任务边界冻结旧 node）从 Adam 层面拦截，是**承重件**——但前提是 A 得先学会。

---

## 5. 请 GPT 裁定的问题

1. **是否认同"v0.6 方向未达成"**——机制启动但 `cos(A,B)<0.5 / disjoint>0` 均 0/6，不能称 Node autonomous specialization？还是判定另有归因？

2. **根本原因裁定**：现在看下来，**A 任务本身学不会（A_curve 起点 0.02，Router 冷启动鸡生蛋）是更底层的问题**。是否应**回到 v0.2 验证过的 warm-up / curriculum（先让 A 学会）**，而非继续加分化机制？请裁定优先级。

3. **freeze_old 该保留还是削弱**：它拦截 Adam 防遗忘（40ep 对比 24ep 证实"没学会"是主因而非"被覆盖"），但 GPT 二审说"不加复杂机制"。freeze_old 是否算可接受的承重件？可更深吗？

4. **下一步建议**（请选/组合）：
   - a) 回到 v0.2 warm-up/curriculum 破鸡生蛋（先让 A 学会，再谈分化）——我倾向此
   - b) 更强模块分化机制（如 Novak 检测器强制 spawn 专属 + 局部训练）
   - c) Dynamic Neural Ecology 进一步（Node 自主决定 specialize/复制/休眠）
   - d) 停止 v0.6，接受"FDN 当前架构无法实现模块分化"的结论

---

## 6. 三层对账清单

| 版本 | commit | 原始 JSON | 报告 |
|---|---|---|---|
| v0.6 批次1-4 | 8351d01 | results/summary_v06_seed{0-5}.json | results/V06_REPORT.md |
| v0.6 批次1 | 0d195b8 | (探针) | EXPERIMENT_LOG (批次1) |
| v0.6 批次2 | dd91ea0 | (探针) | EXPERIMENT_LOG (批次2) |
| v0.6 批次3+3+ | 53d0587 / 4d2a5f3 | (探针) | EXPERIMENT_LOG (批次3/3+) |

---
*（材料由团队按 commit + 原始 JSON + 报告三层对账整理，供外部独立审计裁量下一步方向。）*
