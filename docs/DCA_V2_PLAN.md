# DCA-v2 研究方案：真实数据 + 神经基座（判据先行 + 诚实现实约束）

> **作者**：陈亦安（安安）
> **依据**：GPT 七审（docs/GPT_AUDIT_V10.md）——DCA-v1 过关第一道真正的大门（非 toy-specific）。
> 下一步唯一方向 = **DCA-v2：真实数据 + 稍大神经基座**（小型 Transformer/SSM + 3~5 个真实任务 + DCA Capability Manager），先不接 BDH。
> 终极研究问题：**"什么样的底层神经架构，最适合作为可持续生长 Capability 的载体？"**

---

## 0. 现实约束盘点（判据先行的第一步：现场事实优先）

**环境实测**：
- **无 GPU**（torch cuda=False；8 核 CPU，16G 内存）——GPT 建议的"稍大神经基座"受 CPU 限制。
- **网络受限**（github 超时 / HuggingFace 不可达）——但 **pytorch 官方数据源可达**（实测 MNIST/FashionMNIST S3 返回 200 且成功下载到 `/data/fdn/data`）。
- **torchvision 已装**（0.29），MNIST/FashionMNIST 已真实下载。

**诚实结论**：DCA-v2 用 **pytorch 官方真实数据（MNIST/FashionMNIST）** 作为"真实数据"基础——这是本环境**唯一可得**的真实数据源。神经基座受 CPU 限制，用小型模型（而非"稍大 GPU 基座"），**诚实标注**这一点。

---

## 1. 研究问题（GPT 七审指明）

> **DCA 的"REUSE/SPAWN/FREEZE"机制，能否管理【真实数据】训练出的神经技能？**

DCA-v1 已证明非 toy-specific（合成任务类型）；DCA-v2 升级为**真实数据**（MNIST/FashionMNIST），验证"真实数据训练的技能能否被 DCA 生命周期管理"。

### 关键转折（GPT 七审架构升华）
> DCA **不该规定 Capability 内部是什么网络**（MLP/RNN/SSM/Transformer 都行）。DCA 只管生命周期（出生→学习→冻结→复用→组合→再生）。

所以 DCA-v2 聚焦：**真实技能**（而非网络架构），验证 DCA 管理真实数据技能。

---

## 2. 任务集设计（真实数据，CPU 可训）

基于 MNIST/FashionMNIST，构造**3~5 个真实神经技能**（各含同族变体，测复用）：
- 技能是**图像分类/特征**任务的不同切分视角：
  | skill | 任务本质 | 数据（真实） |
  |---|---|---|
  | `mnist_digit`（数字分类）| 0-9 分类 | MNIST（真实）|
  | `fashion_class`（服饰分类）| 10 类服饰 | FashionMNIST（真实）|
  | `mnist_digit_rot`（旋转数字）| 数字识别+旋转不变性 | MNIST 旋转版本 |
  | `mnist_threshold`（阈值判定）| 按数字奇偶分类 | MNIST（不同目标函数）|
  | `fashion_binary`（服饰二分类）| 上下装二分类 | FashionMNIST |

- **同族变体**：相同任务类型、不同数据分布/切分（如 mnist_digit 的不同子集）→ 测 REUSE。
- **异族技能**：完全不同的任务本质（数字分类 vs 服饰分类 vs 旋转识别）→ 测 SPAWN。

> ⚠️ 这是"真实数据"但 skill 的边界是**人为切分视角**（数字/服饰/旋转/奇偶），仍非完全"开放语义技能"。诚实标注：这是**真实数据 + 受 CPU 限制的神经技能**，是真实数据的起步，非"10B 大模型"。

---

## 3. 核心判据（跨到真实数据的门槛）

### 判据 A：能力边界发现（真实数据上同族复用/异族新建）
- 同族视角（如 mnist_digit 两个变体）→ 复用同一 cap（损伤≈0）。
- 异族视角（mnist_digit vs fashion_class）→ spawn 新 cap（损伤大）。
- **达标**：分化矩阵正确，≥3 seed 一致。

### 判据 B：Interference Predictor 在真实数据技能上预测 damage
- 真实数据的 task_rep（图像统计特征）+ cap_state → 预测 damage。
- **达标**：未见组合上 Spearman≥0.7 / 一致≥0.8 / F1≥0.8。
- **风险**：图像数据集统计特征能否区分技能族（MNIST vs FashionMNIST 的像素统计有差异，但旋转/奇偶变体可能难区分）——**同 DCA-v1 预判的风险**。

### 判据 C：持续学习（真实数据上保旧+学新）
- 学新技能后旧技能不遗忘；新技能学会。
- **达标**：所有技能 acc≥阈值（真实数据阈值需先校准——MNIST 分类可达 0.98+，FashionMNIST 0.85+）。

### 判据 D：多 seed 稳定（≥3 seed）
- **达标**：n_caps / 分化矩阵跨 seed 一致。

---

## 4. 方法框架

### 4.1 Capability（真实技能容器）
沿用 DCA-v1 的**统一输出分类 Capability**（每能力独立路径），输入为图像特征（CNN embedding 后接 MLP 头，或扁平化像素）。
**关键**：沿用 GPT 七审的"Capability 不规定内部网络"精神——cap 内部可以是小型 CNN/MLP。

### 4.2 task_rep（真实数据表征，无 oracle）
从图像数据派生：像素均值/方差/边缘统计/类别分布特征 → 固定向量（不用 skill 名）。

### 4.3 Interference Predictor（跨域到真实数据）
`[task_rep(图像); cap_state]` → MLP → predicted damage。离线真实 probe 生成 ground-truth。

### 4.4 流程
```
真实技能 → task_rep + cap_state → Interference Predictor → predicted damage
  → REUSE（伤小）/ SPAWN（伤大）→ 训练 → FREEZE → Capability Pool
```

---

## 5. 现实约束与诚实边界

1. **无 GPU**：用小型模型（CNN/MLP）+ 缩小数据量（如每类几百样本）跑通机制验证。**不宣称**"大模型已验证"。
2. **真实数据来源有限**：仅 pytorch 官方源可达（MNIST/FashionMNIST）。skill 边界是**人为切分视角**，非完全开放语义技能。诚实标注这是"真实数据的起步"。
3. **能宣称的**（判据达标）："DCA 的 REUSE/SPAWN/FREEZE 能管理**真实数据训练出的神经技能**（图像分类/识别类）——从合成技能跨到真实数据。"
4. **不能宣称的**（GPT 七审红线）："已证明适用于真实大模型/开放语义技能"——需 GPU + 完整真实数据集 + 更好 task_rep。

---

## 6. 分阶段执行（一次只动一组变量）

**阶段1：真实数据技能可学性 + 阈值校准**
- [ ] 5 个真实技能各能被小型 CNN/MLP 学到合理精度（CPU 可跑）
- [ ] 定阈值（mnist_digit≥0.95 / fashion≥0.8）
- **通过 → 阶段2**

**阶段2：能力边界发现（判据 A）**
- [ ] 同族视角复用 / 异族视角 spawn，分化矩阵正确，≥3 seed

**阶段3：Interference Predictor 跨域（判据 B）**
- [ ] 图像 task_rep + cap_state → damage，留出验证 Spearman/一致/F1

**阶段4：持续学习 + 多 seed（判据 C/D）**
- [ ] 保旧+学新，≥3 seed

---

## 7. 达标 vs 不达标处置（事前锚定）

| 结果 | 处置 |
|---|---|
| A/B/C/D 全达标 | 报告"DCA 能管理真实数据训练的神经技能"，DCA 跨到真实数据 |
| B 失败（图像 task_rep 无区分度）| 报告"真实数据技能需更强表征（学习式 task_rep）"，如实记录 |
| A/C/D 部分失败 | 如实记录，探针定位 |

> 方案确认后启动阶段1。数据已确认可得（MNIST/FashionMNIST 在 /data/fdn/data）。
