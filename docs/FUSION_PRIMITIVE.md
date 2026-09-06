# 融合原语方案：稀疏激活 + Fast-Weight 记忆（BDH rho 快权重 + DCA 独立路径）

> **作者**：陈亦安（安安）
> **背景**：爸爸认可——把 BDH 真金的 fast weights（rho 快权重状态）+ DCA 独立路径，融合成"稀疏激活 + 快权重记忆"的原语。
> **依据**：①burn_hatchling 源码证实 BDH 核心真金是 `rho = rho + x_t_latent*v_t`（Hebbian fast weights，状态在边权/突触）——区别 Transformer 的关键；②DCA 已验证"每能力独立路径 + 复用伤旧→新建"（FastWeights v3）+ Interference Predictor 免训决策（29.1µs）。③伪稀疏（稠密GEMM+后置mask）不是出路，已证。

---

## 1. 为什么融合这两条金线

| 机制 | 解决什么 | 已验证 |
|---|---|---|
| **BDH rho 快权重**（`rho += x_latent * v`，Hebbian）| **长上下文记忆**——状态在突触边权上，几百 token 持续 | ✅ burn_hatchling 源码 |
| **DCA 独立路径**（每能力不共享中间表征）| **能力隔离**——新能力不覆盖旧能力 | ✅ FastWeights v3（A/B/A' 全1.0 + A_forget=0）|
| **DCA Predictor**（免训预测 damage）| **训练成本降**——免训决定复用/新建 | ✅ 29.1µs 规模无关 |

**三者组合 = "稀疏激活 + 快权重记忆 + 免训决策"** —— 这正是"区别 Transformer（无持久状态）且能力不输 + 成本双降"的路径。

## 2. 融合架构

```
每个 Capability = 一个 fast-weight 记忆单元：
  慢权重（backprop 学）：encoder(D→N 稀疏维) + decoder(N→D) + head
  快权重（Hebbian 推理期累积）：rho[B,H,N,D]  ← rho += sparse_act[n] * v，状态在边权
  稀疏激活：每 token 只激活 top-k latent 通道（真稀疏门控）
        ↓
DCA 管理层：跨能力按需 spawn/复用（Predictor 免训决策哪个 cap 接新任务）
```

- **稀疏激活**：latent 维 N 每 token 只激活 top-k 通道 → 真稀疏（只算激活通道，非后置 mask）
- **快权重记忆**：rho 在边权上 Hebbian 累积 → 长上下文，无需 KV 全存
- **DCA 管理**：多能力各自独立 rho/权重路径 → 复用伤旧→新建（Predictor 决定）

## 3. 判据（判据先行，跑前锁死）

### 机制判据（真发生）
- **F1 稀疏激活**：每 token 激活 latent 通道占比低（真稀疏，非全集），且**只算激活通道**（不是后置 mask）。
- **F2 快权重记忆累积**：rho 随序列累积（`rho[t+1] != rho[t]`），且长期依赖能力 ≥ 无 rho 版。

### 价值判据（对照 Transformer）
- **F3 能力不输**：字符级语言建模 val ≤ Transformer + 容差（同预算）。
- **F4 推理成本降**：激活参数（稀疏后实际算的）显著 < Transformer 全量，且 wall-clock 更低。

### 组合判定
- **F1+F2+F3+F4 全成立** → 融合原语走通（稀疏+快权重记忆+能力不输+推理降本）。
- 任一失败则如实暴露（如 F4 依赖真稀疏内核，CPU 可能退化为稠密 → 记"CPU 环境验证不出"）。

## 4. 实验设计（一次只动一组变量）
1. 字符级语言建模（同数据）。
2. **对照组**：Transformer（同预算）。
3. **实验组**：融合原语（稀疏激活 + fast-weight rho 记忆）。验证 F1（稀疏占比）/F2（rho 累积）/F3（val）/F4（激活参数+wall-clock）。
4. ≥3 seed。

## 5. 诚实边界（事前锚定）
- **稀疏激活必须"真只算激活通道"**——若只能做到"稠密GEMM+后置mask"（伪稀疏，同 BDH），则 F1/F4 失败，如实记录"CPU 上真稀疏内核不可用"。
- **fast-weight rho 若不能提供长上下文增益**，则 F2 失败——BDH 真金在此环境未兑现。
- **不预设结论**——融合了也不一定赢，诚实暴露失败点，不硬凑。

## 6. 与父目标对齐
父目标 = "找区别 Transformer 的范式，能力不输 + 训练/推理成本双降"。
- **fast-weight 记忆** = 减少长上下文 KV 存储/检索成本（推理降本）；
- **稀疏激活** = 每 token 只算激活通道（推理降本）；
- **DCA 免训决策** = 训练降本。
三者组合才可能真正达标。本实验直接检验。
