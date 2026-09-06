# Dynamic Capability Architecture（DCA）—— 正式立项方向

> **立项依据**：GPT 五审（docs/GPT_AUDIT_V08.md，审计对象 commit `c316651` 验证矩阵全绿）正式裁定：
> "**FDN 可以封存；Dynamic Capability 研究正式开新章**"——GPT 明确"愿意认真押"这条线。
> 本研究替代 FDN（已封存的 MoE 失败路线），是质的转向：从 **Dynamic MoE / Node** → **Dynamic Capability System**。

---

## 0. 为什么改名（FDN → DCA）

FDN 的名称承载着"**已封存的 MoE 专家路由失败路线**"（v0→v0.7 一路 No-Go：MoE 拓扑对 toy 任务梯度分配失败）。
而本方向的研究对象**已经质变**：

```
FDN（旧）：输入 → Router → 选 k 个独立 Node（专家路由）→ 门控加权   [已死：专家学不会/梯度坍塌]
DCA（新）：输入 → Capability Router → 复用/生长 Capability 池 → 输出   [已验证：能力出生/冻结/复用/组合/扩张]
```

**DCA 的核心不是"动态 Node"，而是"能力可以出生、冻结、复用、组合和扩张。"**（GPT 五审原话）
继续叫 FDN 会混淆"已封存的失败路线"与"新立项方向"，故正式改名。

---

## 1. 核心机制（DCA 主循环）

```
新任务
  ↓
尝试复用已有 Capability
  ↓
短暂适应 / Probe
  ↓
检测旧能力损伤（interference）
  │
  ├── 无明显损伤 ──────→ REUSE（复用现有 Capability）
  │
  └── 有明显损伤 ──────→ SPAWN（创建新 Capability）→ 新能力 → FREEZE（冻结）
```

**关键判据（本项目最重要的方法论沉淀）**：
> **"复用旧能力是否伤害旧能力" —— 而不是 "新任务 fit 好不好"。**

为什么：简单任务（ADD/MUL/SUB）一个足够大的 MLP 全部都能学（new-task fit 都很好），
所以"新任务 fit"不能证明应该共享；而 "reuse → old capability degraded" 才真正说明**这两个能力不能共用同一个可塑路径**。

---

## 2. 架构（每个 Capability 完全独立）

```
Capability A          Capability B
├─ encoder            ├─ encoder
├─ computation        ├─ computation
└─ decoder            └─ decoder
```

**能力边界首先是参数隔离边界。** v2 的 shared projection 已证明：Capability A/B 共享中间层，后学 B 会覆盖 A。

*上层结构（Capability Router + Capability Memory + Capability Pool）*
```
           Capability Memory
                │
                ▼
Input ───→ Capability Router ───→ Cap A / Cap B / Cap C ───→ Output
                │
                └── 新能力：interference → SPAWN → new Capability → FREEZE
```

---

## 3. 三条核心原则（c316651 确立，反哺任何动态模块/边权设计）

1. **能力路径必须独立**——每能力完全独立计算路径（encoder+computation+decoder），非共享中间层。
   共享层必被后续任务改写覆盖（v2 实证）。
2. **能力边界不是"相似度边界"**——不能简单 embedding 相似→reuse / 不同→spawn。真正的定义是：
   > **"共享可塑参数是否造成不可接受的 interference。"**（这是能力隔离的本质）
3. **"复用是否伤旧" 比 "新任务 fit" 重要**——simple-task trap：一个足够大 MLP 能全学会所有简单任务，
   new-task fit 好 ≠ 应共享；reuse 造成 old degraded 才是"不能共用一个可塑路径"的铁证。

---

## 4. 验证矩阵（升级门槛，已通过）

**判据先行，一次跑定**：8 个结构化任务 × 5 seed（`/tmp/probe_bdh_matrix.py`，产物 `/tmp/dyn_matrix_results.json`）。

| 判据 | 结果 |
|---|---|
| 分化矩阵正确 | 5/5 seed 完全一致：cap0=[A,A2,A3](加族)/cap1=[B,B2](乘族)/cap2/3/4=[D_sub/C_logic/D_seq] |
| 每任务 acc ≥ 0.85 | 8/8 任务 × 5/5 seed 全达标（多数 1.0，B_mul 最低 0.977）|
| 最终 Capability 数 = 5 | 恰好 = 加/乘/减/逻辑/序列 各 1（非 8=无复用、非 1=无分化）|

**达标 → 正式从 toy mechanism 升级为研究方向。** 无 oracle：模型靠"复用会伤旧→新建"判据，
**自动**把 8 任务分成 5 能力，同类复用（加×3/乘×2 各共用一个 cap）+ 异类新建（减/逻辑/序列各 spawn）。
跨 5 seed 完全稳定 + 8 任务全部 1.0 + 持续学习（加族在学完所有异类后仍全 1.0）。

---

## 5. 诚实边界（GPT 五审踩的重要刹车）

> **这不是"通用全动态智能已经成功"。** 现在能说的是：
> **"在结构化 toy continual-learning 环境中，Dynamic Capability Pool 已证明能稳定地自动发现能力边界、
> 复用同类能力、为异类能力扩容，并保持旧能力。"**

这个结论扎实，但**不宣称**"通用全动态智能"。研究的**诚实边界** = 结构化 toy 持续学习环境。

---

## 6. 下一阶段（GPT 唯一允许：Interference Predictor）

**当前机制的成本痛点**：Probe = 复制 Capability → 真实训练 → 看旧能力受损。
**如果 Capability 是 10B 参数，这东西不能这么干。**

所以下一阶段**不是加 loss、不是上大模型、不是接 BDH、不是 Transformer**，而是研究：

> **Interference Predictor —— 预测 interference，而不用真的训练一次。**

```
现在：Probe → 真实训练 → Damage
未来：Task representation + Capability state → Interference Predictor → 预计 Damage → REUSE/SPAWN
  例：A + A2 → predicted damage 0.01 → REUSE
      A + MUL → predicted damage 0.92 → SPAWN
```

**这是从 toy → architecture 的真正跨越**（研究扩展性/成本/可计算性，而非重复证明"机制有效"）。

---

## 7. 研究命名定案

- **项目名**：Dynamic Capability Architecture（DCA）
- **核心命题**：能力可以**出生（spawn）、冻结（freeze）、复用（reuse）、组合（compose）、扩张（grow）**
- **历史**：FDN（v0→v0.7）作为**已封存的完整反证资产保留**（No-Go / Architecture Study），供后续迁移工具链
  （DynamicNode/telemetry/freeze/continual-learning/router audit）

> 演进路径：`FDN(MoE 专家路由 No-Go) → BDH 研究 → FastWeights v3(每能力独立路径) → 动态池按需 spawn → 验证矩阵全绿 → DCA 正式立项`
