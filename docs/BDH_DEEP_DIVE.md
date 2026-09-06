# BDH 深入分析：能否救 FDN？(BDH_DEEP_DIVE)

> **来源**：官方 `pathwaycom/bdh` + 社区 `ElaMCB/bdh` + 论文 arXiv:2509.26507（The Dragon Hatchling）
> **结论先行**：BDH 的核心机制（**fast weights / Hebbian 边权状态**）与 FDN 的 MoE 专家路由**完全不同**，它印证了 GPT 四审"动态 ≠ MoE 专家路由"的判断。**BDH 思路能救 FDN 的核心心智，但救法是"重构成 BDH 式"而非"修 FDN"。**

---

## 1. FDN 为什么死（一句话复盘）

FDN 把"动态结构"实现为 **MoE 专家路由**：`Router 选 k 个独立 node → 门控加权`。致命伤：
- `topk(scores)` **离散不可导**，只有选中的 node 拿梯度，Router 学不到正确梯度 → 坍塌/专家饿死
- 实验证明：即使 soft 全连通（可导）、加 load-balancing/Z-loss（工业标准件），单任务 A 仍只到 0.05-0.31（StaticMLP=1.0）
- **根因**：把"本来一个共享函数"强行拆给多个独立专家 → 梯度分配失败。**"动态性"放在了错误的位置（选专家）**。

## 2. BDH 的动态性放在哪（关键差异）

**BDH 把动态性放在"边权（突触）"，不是"选专家"。** 三个核心机制：

### (a) Fast weights / 快权重——动态状态在"突触边上"
论文 §1.2 决定性原话：
> "The evolving ruleset can be seen as the temporal state of the reasoning system, sometimes called **'fast weights'** (Hinton & Plaut 1987; Schmidhuber 1993; Ba et al. 2016a). **Fast-weights systems have a favorable ratio of state size to parameter count.**"
- **慢权重**（固定参数）：backprop 学，是网络骨架
- **快权重**（动态连接）：**Hebbian 学习更新，作为推理时的状态**，存在"神经元-神经元边（突触）"上，`A_ij` 表示边权
- 论文 §2.1：状态变量 "may appear both on nodes **and edges**"；"state is larger than number of neuron nodes, appearing on **neuron-neuron edges (synapses)**"

### (b) Edge-reweighting kernel——动态边权重
BDH 是 `edge-reweighting kernel`（边重加权核）：每个局部规则要么是"单节点的计算规则"，要么是"边上的通信规则"。动态性 = **边（突触）权重的持续调整**（Hebbian：`A_ij` 随"共同激活"增强）。

### (c) BDH-GPU = ReLU-lowrank FFN + linear attention（Tensor 化，可训练）
代码 `bdh.py` 的实现本质：
```
x → encoder (D→N=8192 维) → ReLU → 稀疏正激活 (x_sparse)   ← 超大稀疏隐空间
  → linear attention (自相似, K is Q) → x_sparse * y_sparse ← 逐通道乘法门控（连续 soft）
  → decoder (N*n_h → D)
```
- 动态性 = **N 个通道的逐元素乘法门控**（连续、可导），不是离散选专家
- 激活「正 + 稀疏」是设计使然（ReLU 阈值），不是额外正则（论文明确 L1 关闭）
- **规模化参数 `mlp_internal_dim_multiplier=128`**：把共享输入投射到超大稀疏空间，动态性在"通道开合模式"，随 token 变化

## 3. 为什么 BDH 能训、FDN 不能（核心对照）

| | FDN | BDH |
|---|---|---|
| 动态性位置 | **选哪个专家**（离散，Router 选择）| **边权/通道开合**（连续，Hebbian 更新）|
| 梯度 | 只流向选中 node → 坍塌 | **流向所有通道**（连续 gate）→ 稳定 |
| 结构 | 多个独立 node + Router | **单一共享骨架** + 可演化边权 |
| 状态 | node 激活 | **突触边权（fast weights）** 作为推理状态 |
| 可解释 | node 组 | **单个突触定位**到具体概念（monosemanticity）|

**BDH 用"连续逐通道/边权动态"绕开了 FDN 的"离散专家选择"这个致命伤。** 动态性保留（通道开合、边权变化），但**没有离散跳变、没有梯度断点**。

## 4. 关键理论洞察（BDH 告诉我们的）

1. **"动态结构"的正确形态 = 共享骨架 + 快权重（Hebbian 边权状态）**，不是"多专家 + Router 选择"。这**正是 GPT 四审说的**："Node 不一定是专家，可以是计算原语/状态转移"。
2. **"参数/状态比"是关键**（fast weights 优势）：状态存"边"上，`state >> params`，所以能容纳长上下文（working memory 在突触上持续几百 token）。
3. **可解释性**：激活稀疏+正，单突触可定位到具体概念——FDN 一直缺的就是这个。

## 5. 关于"救 FDN"——诚实的三点

### ✅ 能救的是"心智"，不是"代码"
FDN 的**目标**（动态结构、持续学习、快慢分离、可解释）并没有错。错的是**实现方式**（MoE 专家路由）。BDH 证明：**同样追求"动态"，可以不用专家路由**。

### ⚠️ 但 BDH 开源版要泼冷水
- 官方开源 = **论文 baseline 的 GPU 教学代码**（tinyshakespeare 玩具，3000 iter），README 明确声明"**无法复现 97.4% Sudoku / 29.5% ARC**"。
- **Sudoku / ARC / BDH-CQ 核心实现、训练管线、数据集、权权重**——官方都没开源。我们学到的只是**架构骨架**，不是它们宣称的能力。
- BDH 是**语言建模**任务（token 序列），我们的 FDN 是 toy 回归。**BDH 机制要在回归任务上验证，需重新移植**。

### ❌ 关键：BDH 的"动态"是否真能解决"持续学习 / 模块分化"？
BDH 的 fast weights 是**推理期的 Hebbian 状态**（working memory），不是**训练期的模块分化**。FFDN 真正想要的是"新增能力 → 新结构"。BDH 的"动态结构"更多是"**同一骨架内通道开合**"，**没有证据表明它能像 FDN 期望的那样"spawn 出新模块"**。这是 BDH 与 FDN 目标的**根本错位**——BDH 解决的是"如何用共享骨架处理长上下文/可解释推理"，FDN 想要的是"如何按需生长模块"。

---

## 6. 结论

**BDH 不能直接"救活" FDN**，因为：
1. BDH 的动态性 = **共享骨架内的连续通道/边权开合**（解决"长上下文 + 可解释推理"），FDN 想要的是"**按需生长离散模块**"（解决"持续学习 + 模块分化"）——**目标不同**。
2. BDH 开源版是 baseline 教学代码，Sudoku/ARC 能力不可复现，无权重无训练管线。

**但 BDH 给了 FDN 最重要的启示**：
- ✅ 动态性**不该放在"选专家"**——放"边权/通道开合"（连续可导）才不塌
- ✅ "动态结构"可以**没有 MoE 路由**（共享骨架 + 快权重）
- ✅ GPT 四审说的"State-Space / Dynamic-State / Self-Modifying Network"——**BDH 就是这个方向的实证**（论文自称 "attention-based state space sequence learning architecture"）

**真正的下一步**是：把 BDH 的**关键机制**（共享骨架 + 连续稀疏通道门控 / fast weights 边权状态）**移植到我们熟悉的 toy 回归任务**，做**最小反证**：
- 单任务 A：共享骨架 + 逐通道门控能否学到 1.0？（对比 StaticMLP=1.0）
- 若成立 → 这就是"动态结构不是 MoE"的**可训、不塌、可解释**新架构，正是 GPT 指的方向。

> 完整论文已存 `/root/.hermes/cache/web/arxiv.org-57f3b060ce.md`（50,228 字符）；社区 fork 在 `/data/bdh_stateful/`。
