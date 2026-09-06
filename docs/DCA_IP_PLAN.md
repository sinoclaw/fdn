# Interference Predictor 研究方案（判据先行）

> **背景（GPT 五审，docs/GPT_AUDIT_V08.md）**：DCA 已立项（Dynamic Capability Architecture），
> 当前机制用 `复制 Cap → 真实训练 → 测旧能力损伤` 决定 REUSE/SPAWN。
> **成本痛点**：若 Capability 是 10B 参数，真实训练 probe 一次不可接受 → 无法扩展。
> **下一阶段唯一允许**：Interference Predictor —— 预测 interference，而不用真的训练一次。
> 这是从 toy → architecture 的真正跨越（研究扩展性/成本/可计算性）。

## 问题定义

把 DCA 的决策点 `Probe(reuse cap c on task t) → real training → damage` 替换为：

```
task representation r_t  +  capability state s_c  →  Interference Predictor  →  predicted damage  →  REUSE / SPAWN
```

其中 `damage =` cap c 复用学习任务 t 后，其对**旧任务**的 acc 下降程度（interference）。

## 为什么这有研究价值（GPT 原话）

- 现在：`Probe → 真实训练 → Damage`（贵）
- 未来：`Task representation + Capability state → Interference Predictor → 预计 Damage → REUSE/SPAWN`（免训练）
- 例：`A + A2 → predicted damage 0.01 → 复用`；`A + MUL → predicted damage 0.92 → 直接 spawn`，**连试错训练都不用**。

## 关键方法论约束（诚实性）

1. **无 oracle task-id**：task representation 必须是"可泛化"的——从**输入数据本身**提取的统计量
   （均值/方差/min/max/分布形状/特征相关性），而非人为标好的任务编号。否则就是 oracle，研究价值归零。
2. **ground-truth 只能来自真实 probe**：预测器的监督信号 = 用真实训练 probe 测出的 damage（离线一次性生成）。
   预测器学的只是"从表征映射到 damage 的近似"，**推理时不再需要真实训练**。
3. **留出验证是关键**：必须用**训练时没见过**的 (cap, task) 组合测泛化——否则是记忆不是预测。
   - 训练集 / 验证集按任务类型划分：如加+乘+减 训练，逻辑+序列 留出测（未见过的能力组合）。
4. **判据先行**：达标才继续。

## 特征设计（无 oracle）

**task representation `r_t`**（从输入数据派生，不依赖任务名）：
- 输入 x 的逐维统计：均值/标准差/min/max/偏度
- 输入分布形状：输入对之间的相关系数
- 目标 y 的分布统计（均值/方差/min/max）
- 输入-目标关系的可预测性信号（如线性回归残差——但注意不能偷看真实的函数求法）
- → 一个固定长度向量，同一任务无论 seed 都稳定（统计量是分布属性）

**capability state `s_c`**：
- cap 的 encoder 权重统计：权重 norm / gate 激活稀疏度（活跃通道比例）/ 输出分布
- cap 训练过的任务集合的 r_t 的平均（cap 的"累积表征"）
- → 固定长度向量

**predicted damage target**：
- 真实 probe：拿 cap c 的一个深拷贝，用任务 t 训练 N epoch，测 cap 对**它原任务**的 acc 平均下降
  = `damage(c, t)`（标量 ∈ [0,1] 或 acc 绝对差）

## 预测器

- 简单模型：`[r_t; s_c]` 拼接 → MLP(1-2 层) → 标量 predicted damage（回归）
- 也可做二分类：predicted damage 是否 > 阈值（REUSE vs SPAWN）
- **判据先行**：见下

## 判据（升级门槛）

1. **Damage 预测准确性**：留出（未见过的任务类）组合上，`predicted damage` 与 `ground-truth damage`
   的 **Spearman 相关 ≥ 0.7**（排序正确性——决定"哪个任务更伤"）。
2. **决策一致性**：预测器输出 `REUSE/SPAWN` 与真实 probe 的 `REUSE/SPAWN` 一致率 ≥ 0.8（F1 ≥ 0.8）。
3. **多 seed 稳定**：≥3 seed 结果一致（非单 seed 偶然）。
4. **可泛化**：留出的任务类型（训练没见过）能力边界判断正确——如逻辑/序列任务在新出现的 cap 上
   被预测为"高 damage → 应 spawn"，而加变体被预测为"低 damage → 应 reuse"。
5. **诚实边界**：只宣称"在结构化 toy 持续学习 + 未见能力组合上，Interference Predictor 能免训练预测
   damage 以指导 REUSE/SPAWN"——不宣称"通用全动态智能"。

## 允许 / 禁止（GPT 五审红线）

- ✅ 允许：Interference Predictor（扩展性/成本/可计算性）
- ❌ 禁止：马上上大模型 / 接 BDH / 上 Transformer / 加 loss 继续磨 DCA 有效性
  （c316651 已证明机制有效，继续证明同一件事没有意义）

> 实现：`/tmp/probe_ip.py`；结果存档 `results/EXPERIMENT_LOG.md` + `results/DCA_IP_REPORT.md`。
