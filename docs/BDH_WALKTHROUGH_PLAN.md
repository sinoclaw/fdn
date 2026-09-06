# BDH 机制走通验证方案：BDH vs 等参数量 Transformer（能力不输 + 成本双降）

> **作者**：陈亦安（安安）
> **目标**（爸爸拍板）：验证 BDH 范式（fast weights + 稀疏门控）**能不能走通**——走通的定义 = **同一任务、等参数量下，BDH 能力 ≈ Transformer（不输），且训练+推理成本显著更低（双降）**。
> 走通后下一步：找对标模型 → 蒸馏 → 和它对打。

---

## 1. 关键认知（先立诚实边界）

- **官方开源 `bdh.py` 是论文教学 toy 代码**（tinyshakespeare、3000 iter、256维、512 block），跑不出论文宣称的 Sudoku 97.4% / ARC 29.5%（核心 BDH-CQ / 训练管线 / 权重未开源）。
- **因此"走通"定义为机制级**：验证 BDH 的**架构骨架**（K is Q 因果注意 + 逐通道稀疏乘法门控 + 超大稀疏维）在**同等条件下**能否达到 Transformer 级能力，且成本更低。
- **不声称**复现 Sudoku/ARC 数字——那是营销话术产物，本环境无权重无管线。

## 2. BDH 核心机制（`bdh.py` 关键点）

```
x → embed(D) → encoder(D→N=mlp_internal_dim_multiplier*D//nh 超大稀疏维, 默认8192)
  → ReLU → x_sparse（稀疏正激活）
  → Attention(K is Q, 因果 tril) → yKV
  → yKV → encoder_v → ReLU → y_sparse
  → xy_sparse = x_sparse * y_sparse（逐通道乘法门控，连续可导）
  → decoder(N*nh → D) → ln → 残差
```
- **动态性在"通道开合"（连续逐元素乘法门控）**，不是离散选专家 —— 所以可导、不塌、稳定训练。
- **K is Q（`assert K is Q`）** —— 自相似结构。
- **超大稀疏隐空间（N=8192）** —— 稀疏正激活。

## 3. 判据（判据先行，跑前锁死）

### 判据 A：能力不输
同一字符级语言建模任务（tinyshakespeare 或本地构造）、等参数量下，训到相同预算（iterations），对比 val loss：
- **BDH val loss ≤ Transformer val loss + 容差（如 ≤ +0.05）** → 能力不输 ✓
- 更强判据：perplexity 对比；或生成质量（抽样对比）。

### 判据 B：成本双降
| 维度 | 指标 | 达标 |
|---|---|---|
| **训练成本** | 达到目标 val loss 的 wall-clock / FLOPs | BDH ≤ Transformer（时间或 FLOPs 更低）✓ |
| **推理成本** | 单步 forward 的 wall-clock / FLOPs / 峰值显存 | BDH ≤ Transformer ✓ |

> 注意：BDH 有超大稀疏维（8192），参数可能更多，但**稀疏激活意味着实际计算量可能更低**——需实测 FLOPs 或 wall-clock 判断，不预设结论。

### 组合判据（走通 = A 且 B）：
**A（能力不输）且 B（成本双降）同时成立 → BDH 机制走通。**

## 4. 实验设计（一次只动一组变量）

- **任务**：字符级语言建模（tinyshakespeare，官方 train.py 用；github 被墙则本地构造等量字符语料）。
- **对照组**：等参数量 Transformer（nanoGPT 结构：embed + 多层 masked attention + MLP + lm_head），参数尽量对齐 BDH。
- **变量**：只变架构（BDH vs Transformer），其余（数据集、block size、batch、iters、lr、seed、参数量目标）固定。
- **规模**：CPU 可跑（n_layer 小：如 2-4 层，n_embd 128-256，N 维度适中）。
- **多 seed**：≥3 seed 取均值（判据先行铁律）。

### 分步：
1. **数据准备**：准备字符级语料（本地构造，避免 github）。
2. **基线跑通**：确认 BDH 官方实现能在 CPU 上训练（loss 下降），无崩溃。
3. **等参数量对照**：构建同参数量 Transformer，同预算训练。
4. **测 A**：val loss / perplexity 对比。
5. **测 B**：训练 + 推理 wall-clock / FLOPs / 显存对比（用 torch profiler / FLOPs 计数）。
6. **多 seed**：≥3 seed，取均值定判据。

## 5. 诚实边界（事前锚定）

- 若 **A 不成立**（BDH 能力明显输）→ 报告"BDH 机制级能力不达标"，如实记录，不硬凑。可能原因：超大稀疏维在字符级 toy 上优势不显、训练预算不足、架构未调优（但"一次只动一组变量"原则下不做过度调参）。
- 若 **A 成立 B 不成立**（能力 ok 但成本不降）→ 报告"BDH 能力不输但成本优势未实现"，需分析（可能稀疏激活在 CPU 上未充分利用，或参数过多压过稀疏收益）。
- 若 **A 且 B 都成立** → **BDH 机制走通**，进入下一步（找对标模型 / 蒸馏）。

## 6. 达标的下一步（预锚定）
走通后：① 选一个开源小模型作为对标（如 GPT-2 级 / 某小 LLM）；② 蒸馏 BDH 到该任务集；③ 能力和成本对打。

> 方案确认后启动：数据准备 → BDH 基线跑通 → 等参数量 Transformer 对照。
