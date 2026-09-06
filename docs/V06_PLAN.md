# FDN-v0.6 执行方案 — Dynamic Neural Ecology（Node 自主专化）

> 依据外部 GPT 二审判定（docs/GPT_AUDIT_V05.md，commit b25889f）：**撤销 w=0.5 稳定结论、当前不能称 Dynamic Neural Ecology、下一步 = Node autonomous specialization。** v0.6 = Novelty-triggered Spawn + Node Competence + Plasticity Decay。
> GPT 关键洞察：**「Router 负责发现已有能力；Node Ecology 负责产生新能力。」** 现在缺的不是 Router 怎么选，而是 **Node 为什么不产生 specialization pressure**。
> **爸爸拍板**：方向对，就是工程化问题。本方案把 GPT 的机制主张落成可测的工程实现。

---

## 0. 一句话目标

**把「任务分化」从「Router 的职责」改为「Node 的自主行为」——Node 自己观察 novelty/competence，低了就 spawn 新 Node 去专攻某个新输入模式，成熟后塑性衰减保护已有能力。** 用最简 A→B→A' 序列验证「Node 能自主形成 Group A / Group B / Group A'」。

---

## 1. 判据先行（跑任何代码前锁定，GPT 指定）

**最简实验**：`A → B → A'`，A、B **难度相同**（都用 gen_add/gen_sub 同级，不用布尔 logic）。
**多 seed**（至少 5，本方案 6）——从 v0.5 学到的铁律：**单 seed 高分（seed3 A=0.973）不算证据，必须多 seed + specialization + 能力三件套一起过。**

| 指标 | 通过阈值（5/6 seed 达标） | 意义 |
|---|---|---|
| A_end | > 0.4 | A 学会并保留 |
| B | > 0.3 | B 学会（v0.5 里 B 全 <0.1，这是核心提升指标） |
| A2 | > 0.35 | A' 调用 A 的 Node |
| forgetting_A | < 0.05 | A 不遗忘 |
| **cos(A,A')** | > 0.7 | A/A' 复用（同一 Group） |
| **cos(A,B)** | < 0.5 | A/B 分化（不同 Group） |
| **disjoint(A,B)** | > 0.0 | A/B 有不相交 Node |

**判定**：达标 → 「Node 自主专化」机制成立，进入 v0.6 完整版 A→B→D→A'。未达标（尤其 cos(A,B) 仍 >0.5 / disjoint=0）→ 说明 Node 仍未产生 specialization pressure，须诊断 spawn 判据而非加复杂度。

---

## 2. 机制设计（GPT 主张的工程化落地）

### 2.1 关键转变：从「全局新颖度 spawn」→「Node 级 competence-gap spawn」

**现状（错）**：controller 的 spawn 用**全局** novelty = `query 与所有 node key 的 min cos < 阈值`（`_best_key_cos`）。这是"Router 视角"——看整个输入分布离所有 node 都远才 spawn，导致 A/B 都挤到已有 node，无分化。

**v0.6（对）**：**每个 Node 记录自己在当前输入模式下的 competence**，spawn 由「所有被选中 Node 对该模式 competence 都低」触发——这才是"Node 视角"。

### 2.2 给每个 Node 增加的状态（Node 自治）

在 `model/dynamic_node.py` 的 DynamicNode 内（或 fdn.py 的 buffer，跟 v0.3 的 competence 类似，但**按输入模式**而非全局）：

```
per-Node 状态：
- competence（当前模式下的能力）   ← 已有（EMA），但要改成「该 Node 最近被投喂的输入模式的预测误差」驱动，而非全局熵
- novelty（对当前输入的新颖度）    ← 新增：当前输入 query 离该 Node key 的距离（1 - cos）
- reward（该 Node 输出的增益）     ← 新增：该 Node 是否降低了整体 loss
- usage                       ← 已有
- plasticity（可塑）            ← 已有（v0.3 只按 maturity 衰减）
- age / life_stage             ← 已有部分（controller 有 age）
- specialization vector        ← 用 node_key 本身作为"我擅长什么"的表示
```

### 2.3 Novelty-triggered Spawn（核心一刀）

**spawn 判据 = f(novelty, error, competence, usage, capacity)**，是所有激活 node 的**加权**综合，不是"loss 高就 spawn"：

```
对当前 batch 激活的节点集 {i}：
  avg_competence = mean(competence_i)         # 现有能力对这个模式够不够
  avg_novelty    = mean(1 - cos(q, K_i))       # 这个模式新不新
  error_high     = mean(prediction_error_i) > 阈值   # 现有节点是不是做得差
  sustained      = 该模式持续了 ≥ 若干 batch（非噪声）
  capacity_ok    = node_count < max_active

  spawn = (avg_competence < c_thr) AND (avg_novelty > n_thr)
          AND error_high AND sustained AND capacity_ok
```
- 满足 → `_append_node(parent=最高 novelty 的 node, noise)` 长出**专攻该模式的子节点**，并 `add_param_group` 加入优化器（本仓已处理）。
- 新 Node 初始 optimization：**只让它学这个新模式**（先给它专属的软门控——回退 v0.3 的 per-node warm，新 Node 高门控独立吸梯度）。

### 2.4 Plasticity Decay（Node 自寿）

GPT 生命周期：`年轻(高可塑) → 成熟(低可塑) → 休眠(极低) → 重新激活(临时提高)`。
v0.3 已有雏形（maturity 驱动衰减），v0.6 补全到 **5 态**：

```
NEW → WARMING → LEARNING → MATURE → DORMANT → REACTIVATED
plasticity：NEW 高 → MATURE 低（×0.1/ep）→ DORMANT 极低(×0.01) → REACTIVATED 临时×5
```
- 触发：maturity 达阈值→MATURE；长时间低 usage→DORMANT；Router 重新高分命中→REACTIVATED。
- 落点：`update_lifecycle` 的 plastic 乘子改为**按 5 态分级**（现在只有 mature×0.1 一档）。

### 2.5 Node 自主 specialization（"我擅长什么"作答）

Node 的 specialization 用 **node_key 向量** 表示——它在 Router 空间里的位置。当 Node 学会某模式，其 key 靠近该模式的 query；相邻 Node key 高相似 → merge（已有）。**关键新增：Node 观察自己的 usage/reward 占比，高的继续巩固（plasticity 保高直到成熟），低的让位（plasticity 降/休眠）**。

---

## 3. 执行步骤（分批、每批可回退、一次只动一组变量）

### 批次 1：Node 级 competence / novelty 状态落地（纯测量，不改训练）
- `model/fdn.py`：给每 Node 加 `node_competence`（EMA，用**该 node 被选中时的 prediction error** 驱动）、`node_novelty`（1-cos(q,k)）；在 forward 里选中 node 时更新。
- 只**记录**，先看：A/B/D 训练时每 node 的 competence/novelty 分布 → **确认"现有 node 对 B 是否 competence 低"**（验证 GPT hypothesis 成立）。
- **回退**：只加 buffer + 记录，不改路由，可直接弃。

### 批次 2：Novelty-triggered Spawn（改 controller，核心）
- `growth/controller.py`：`task_boundary` 的 spawn 改为**Node 级综合判据**（2.3），替代全局 `_best_key_cos`。
- 加 `compute_node_competence/novelety` helper。spawn 记录 `spawn_reason`（novelty/competence/error 各值）供诊断。
- **回退**：spawn 分支是 condition 单独一段，可开关；关掉回到全局模式。

### 批次 3：Plasticity Decay 5 态（改 lifecycle）
- `model/fdn.py`：`update_lifecycle` 的 plastic 乘子按 `NEW/WARMING/LEARNING/MATURE/DORMANT/REACTIVATED` 分级；加 `life_stage` buffer。
- **回退**：只在现有 plastic 衰减上加分级，参数默认回到 v0.3。

### 批次 4：最简实验 A→B→A'（多 seed）
- `experiments/tasks.py`：定义 `MINI_SEQUENCE = ["A_add","B_gen","A2_add"]`（B 用同级难度，可用 gen_sub 或另造一个同级 B'）。
- 跑 6 seed，判定走第 1 节判据表。产出 `results/summary_v06_seed{0-5}.json`。

### 批次 5：报告 + 归档 + 二审
- `results/V06_REPORT.md`（判据逐条 + specialization matrix + 多 seed）+ EXPERIMENT_LOG + README + git 提交推送。
- **交 GPT 三审**：是否 Node 自主专化已达成（cos(A,B)<0.5 / disjoint>0 + B>0.3）。

---

## 4. 时间/成本预估

| 批次 | 内容 | 耗时 |
|---|---|---|
| 1 | Node 级状态落地 + 探针看分布 | ~30min |
| 2 | Novelty-Spawn 改 controller | ~40min |
| 3 | Plasticity 5 态 | ~30min |
| 4 | 多 seed 实验 A→B→A'（6 seed 并行） | ~30min |
| 5 | 报告 + 提交 | ~20min |
| **合计** | | **~2.5h** |

---

## 5. 明确不做（GPT 三令五申）

- ❌ **不加 task embedding oracle**（GPT：会变成"给 Router 加强 oracle 换漂亮矩阵"，研究价值下降）。
- ❌ **不碰 w**（撤销 w=0.5 结论）。
- ❌ **不接 BDH**。
- ❌ **不一次搞 A/B/C/D**（GPT：v0.6 只 A→B→A'）。
- ✅ **一次只动一组变量**（批次 1/2/3 是三个独立可回退模块）。
- ✅ **多 seed + specialization + 能力三件套**（v0.5 的血泪教训）。

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| Novelty-Spawn 误触发（噪声 spawn） | spawn 加 `sustained`（连续 N batch 高 novelty）+ `error_high` 双门槛，拒绝单样本触发 |
| 新 Node 学了 A 不学 B（还是挤一起） | 批次 2 spawn 后新 Node 用 **per-node warm**（v0.3 已验证）独立吸梯度，避免和旧 Node 共享 |
| spawn 爆炸 | `max_active` 上限 + `capacity_ok` 守卫 + spawn 后冻结期（controller 已有 freeze_tasks） |
| Node 阈值难调 | 用探针（批次1）看实际 competence/novelty 分布定阈值，不盲设 |
