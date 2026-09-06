# GPT 三审对 FDN-v0.6 的裁定（audit commit 523a381）

> **审定对象**：commit `523a381`（v0.6 完整结果：Node 级 spawn + Plasticity 5 态 + freeze_old + 多 seed）
> **来源**：ChatGPT 分享 `t_6a9d17428a748191af3bb3bd746c0f4e`（@oai-dm=1 抓到正文）
> **结论**：**v0.6 未实现「Node autonomous specialization」**——真正的根因被精确定位。

---

## GPT 裁定全文

老6，**`523a381` 这次我认真审了。这个 commit 的价值很高，因为它把 v0.6 的"机制有没有真的成立"扒开了。**

我的裁定和仓库自己提出的问题基本一致，但我会再往前推一步：

### 结论：v0.6 没有实现"Node autonomous specialization"

而且现在已经能比较明确地定位原因了。

### v0.6 实际发生了什么

你们已经把三个机制都真正启动了：

- Node-level competence / novelty
- novelty-triggered spawn
- Plasticity 5-state
- `freeze_old`

甚至确实触发过 spawn：例如 node12 的 novelty/error 都很高，spawn score 达到 0.512。

但是核心指标：
```
cos(A,B) < 0.5       0/6
disjoint(A,B) > 0    0/6
```
而且 **6 个 seed 的 cos(A,B) 全部 > 0.988**。

这不是"分化不够"。这是：**根本没有发生分化。**

所以我同意：**v0.6 方向当前未达成。**

---

### 但真正重要的是：别继续加"分化机制"

这一点我和 Hermes 的倾向有一点不同。

仓库现在发现：**A 本身很多 seed 就没学会。**
```
seed 0: A_end 0.021
seed 2: A_end 0.002
seed 5: A_end 0.002
```

然后你再要求：`A → 专化；B → 另一批 Node`。其实没有意义。因为：
```
A 没形成能力
    ↓
没有稳定 competence
    ↓
novelty / routing 信号不可靠
    ↓
spawn 出来的东西也没有明确归属
```

所以我会把问题分成**两个阶段**。

---

### 第一阶段：先解决 Router Cold Start

这里应该**回到 v0.2 已经验证过的 warm-up/curriculum**。但不是简单把 v0.2 原样搬回来。

我建议做：
```
                NEW TASK A
                    │
                    ▼
              Soft Warm-up
                    │
              Router 学习
                    │
             Node 获得能力
                    │
                    ▼
             Competence > T
                    │
                    ▼
               A LOCKED
                    │
                    ▼
              Hard Routing
```

**只有 A 真正学会之后，才进入 Dynamic Ecology。**

验收条件不要用 epoch，而用 competence/performance：
```
A_accuracy > threshold
AND routing entropy < threshold
AND 连续 N 次稳定
```
满足以后才：`A → MATURE`。

---

### 第二阶段才测试 Node 分化

然后：
```
A learned
    ↓
A nodes protected
    ↓
B arrives
    ↓
existing A nodes competence(B) low
    ↓
novelty/error high
    ↓
spawn
    ↓
B gets new Node
```

这时候才有意义。整个实验应该变成：
```
A
│
├── warm-up
├── learn
└── mature
        │
        ▼
       B
        │
        ├── A nodes protected
        ├── competence gap detected
        └── spawn B node
                 │
                 ▼
               B learn
                 │
                 ▼
               A'
                 │
                 ▼
             reactivate A
```

---

### `freeze_old` 要不要保留？

**保留。** 但我要给它重新定位：

> `freeze_old` 不是 Dynamic Ecology 的核心创新，而是一个实验基线/承重件。

因为你们已经证明：
```
没有 freeze_old → Adam 可以继续改旧 Node → 遗忘
有 freeze_old → 可以保护已经形成的能力
```

而且仓库自己已经发现，Plasticity Decay**只控制 Hebbian 推理期更新，并没有阻止 Adam 的主训练梯度**；真正能从 Adam 层面阻止覆盖的是 `freeze_old`。

所以现在：**不要删。** 但也不要把它包装成"全动态解决方案"。

---

### 还有一个特别重要的坑：你们现在的 cos 指标已经失效了

这个我觉得是 `523a381` 最大的发现之一。

你们现在看到：
```
cos(A,A') ≈ 1
cos(A,B) ≈ 1
```
以前可能会解释成：A 的 Node 被 A' 成功复用了。现在已经知道不是。真正原因是：
```
A ─┐
B ─┼──→ 同一批 Node
A'─┘
```
所以 cos(A,A') ≈ 1、cos(A,B) ≈ 1 自然全部接近 1。

**这个指标以后不能单独拿来证明 specialization。**

---

### 我建议 v0.7 直接这么做

别再继续 v0.6 打补丁。

## FDN-v0.7 = Warm-up → Mature → Protected → Novelty Spawn

只做 **A → B → A'**。

**Phase 1:**
```
A
↓
soft routing
↓
直到 A competence 达标
↓
MATURE
```

**Phase 2:**
```
B
↓
A Node protected
↓
B competence gap
↓
Novelty-triggered spawn
↓
新 Node 学 B
```

**Phase 3:**
```
A'
↓
Router 找 A
↓
A Node reactivation
```

然后看：

| 指标 | 目标 |
| --- | --- |
| A 初始 | > 0.7 |
| B | > 0.5 |
| A' | > 0.5 |
| Forgetting | < 0.05 |
| A/B route overlap | **明显下降** |
| B 是否产生新 Node | **是** |
| A' 是否回到 A Node | **是** |

这里我甚至不建议一开始要求 `cos(A,B)<0.5`。**先要求"B 确实产生了新的 Node"，再谈 specialization 强度。**

---

### 最关键的一点

老6，现在这个项目已经到了一个非常好的研究节点：
```
v0       结构动态失败
 ↓
v0.1     结构稳定
 ↓
v0.2     发现 Router cold-start
 ↓
v0.3/v0.4 发现保护旧能力有效
 ↓
v0.5     发现 specialization 没出现
 ↓
v0.6     证明：
         「novelty + spawn + plasticity」
         如果没有稳定的初始能力，
         仍然无法自动形成模块
```

所以现在真正的研究问题已经变成：

# **"能力形成"是不是动态结构形成的前置条件？**

我认为答案大概率是**是**。

也就是说，我们原来想的是：**先动态长结构 → 再长出能力。**

现在实验逼着我们考虑：**先形成稳定能力 → 再根据能力差异动态分裂结构。**

这个变化非常重要。

**所以我投 `a + freeze_old`，暂时不投 b/c/d。**

先把：**Warm-up → Competence → Freeze → Novelty Spawn** 这个闭环跑通。

如果 v0.7 仍然无法产生 A/B 分化，那时候才值得认真怀疑：FDN 当前的 Router/Node 架构本身就没有产生模块分化所需要的梯度压力。那就不是继续调参了，而是该改核心架构。

---

## 裁定要点（提炼）

1. **🔴 v0.6 未达成**：cos(A,B) 全 >0.988、disjoint 全 0——**根本没发生分化**。
2. **🟢 根因定位**：A 本身没学会（seed0/2/5 的 A_end≈0.002）→ 无稳定 competence → 信号不可靠 → spawn 无明确归属。
3. **🟢 核心研究问题翻转**：「**先形成稳定能力 → 再根据能力差异动态分裂结构**」vs 原假设「先动态长结构 → 再长出能力」。
4. **🟢 分两阶段**：第一阶段先解决 Router Cold Start（v0.2 warm-up/curriculum，用 competence 而非 epoch 验收）；第二阶段才测 Node 分化。
5. **🟢 freeze_old 保留**，但重新定位为**实验基线/承重件**（拦截 Adam 对旧 node 的覆盖），不是 DNE 核心创新。
6. **⚠️ cos 指标失效**：A/B/A' 共用同一批 node 导致 cos 全 ≈1，**不能单独证明 specialization**。
7. **🟢 投 `a + freeze_old`**：不做 b(强分化)/c(DNE 深入)/d(收尾)。

## 下一步：FDN-v0.7 = Warm-up → Mature → Protected → Novelty Spawn

- 只做 `A → B → A'`
- 验收：A>0.7 / B>0.5 / A'>0.5 / forgetting<0.05 / A/B overlap 明显下降 / **B 确实产生新 Node** / **A' 回到 A Node**
- **暂不要求 cos(A,B)<0.5**，先证「B 产生了新 Node」再谈 specialization 强度
