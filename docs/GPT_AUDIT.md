# FDN-v0 外部 GPT 审计回复（2026-09-05）

> 由 ChatGPT 外部分享对话转 Markdown。审计对象：`github.com/sinoclaw/fdn`（FDN-v0 实验仓库）。
> GPT 结论：v0 核心假说未被支持；建议改做 **FDN-v0.1 Stability Patch**，一次只动一个变量。

---

#### ChatGPT said:

行，我刚重新把你这个仓库 **`sinoclaw/fdn`** 看了一遍。这个版本比我刚才描述的还清楚：仓库已经把六维动态和实验判据都落到代码/文档里了，而且 README 明确承认 v0 的核心假说目前 **没有被支持**。[GitHub](https://github.com/sinoclaw/fdn)

我建议现在 **不要大改架构**，直接做 **FDN-v0.1 Stability Patch**。

### 我会让 Hermes 先改这 5 件事

**1. Spawn 冻结期**

新 Node 创建后：

```

spawn
  ↓
warm-up N steps
  ↓
禁止 prune / merge
  ↓
允许正常学习
```

否则刚长出来就可能被结构控制器干掉。

**2. Merge 改成“长期稳定相似”**

现在是：

```

cos(key_i,key_j) > 0.95
→ merge
```

太激进。

改成：

```

similarity > threshold
AND
连续 N 次满足
AND
两个 Node 都不是 young
AND
被合并 Node 的 usage 足够低
```

**3. Evolution 降频**

不要每个训练阶段疯狂：

```

spawn → merge → spawn → merge
```

改成：

```

正常训练
   ↓
积累统计
   ↓
Evolution step
   ↓
冻结
   ↓
继续训练
```

**4. 给 Node 加年龄和稳定度**

每个 Node 至少记录：

```
Python

age
usage_ema
loss_contribution_ema
last_spawn_step
last_mutation_step
stability
```

Evolution Controller 根据这些数据做决定。

**5. 最重要：先关掉 Hebbian plasticity**

这一条我反而最建议做。

先跑：

```

FDN
├── Dynamic Router
├── Dynamic Structure
└── Static Node Weights
```

如果这样已经能证明：

> A → B → C → A' 能调用原来的 Node，并保持 A 能力

再打开：

```

Hebbian plasticity
```

否则现在有三个变量同时在动：

```

结构
+
路由
+
权重
```

实验根本不好解释。

* * *

### 为什么我这么判断？

现在的数据非常有意思：

**A → A' Node IoU = 1.0**

说明 Router 已经成功学到了“同一类任务回来，重新找原来的 Node”。但最终 FDN 有 **54 个 Node**，发生了 **42 次 spawn + 5 次 prune + 42 次 merge**，而 A2 只有 **0.092**。也就是说：

> **“记住在哪里”已经成功；“记住怎么做”失败。**

这两个问题一定要拆开解决。[GitHub](https://github.com/sinoclaw/fdn)

而且仓库自己给出的判定标准也是合理的：需要同时看遗忘、A/A' Node overlap，以及 A' 能力是否接近初始 A。[GitHub](https://github.com/sinoclaw/fdn)

所以我建议 **v0.1 只回答一个问题：**

> **一个已经学会任务 A 的 Node，在经历 B、C 后，能不能被 A' 重新调用，并且仍然会做 A？**

如果答案是 YES，咱们再把 **plasticity → dynamic τ → memory → dynamic compute** 一个一个打开。

这次别让 Hermes 一口气改十个东西。**一次只动一个变量。**这才不会再掉进“看起来全动态，最后不知道到底是哪一层坏了”的坑里。😎

直接查看 FDN 仓库

* * *
