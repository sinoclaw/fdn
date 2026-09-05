# FDN-v0.2 外部 GPT 审计回复（2026-09-06，链接 t_6a9c8ac0）

> ChatGPT 外部分享对话转 Markdown。审计对象：`github.com/sinoclaw/fdn`（commit `8a2b24a` = FDN-v0.2，含 v0.2-A/B/C warm-up 三档 + soft→hard curriculum）。
> GPT 结论：**Router warm-up 有效（v0.2-B 最有希望）；但「每任务重新 soft warm-up」是错的（v0.2-C 灾难性遗忘）。下一步 = FDN-v0.3 Protected Expert Formation**（新模块 soft、旧模块 protected），并指出「Dynamic Neural Ecology」方向。

---

#### ChatGPT said:

哈哈， **8a2b24a 我看到了**。这次是真的到 **v0.2** 了，而且结果很有意思。

我先纠正一下我上一轮的判断：

> **v0.2 已经证明"Router cold-start"确实是核心问题之一，但也同时证明了"soft routing"不能简单地一路软到底。**

### v0.2 的三个实验非常关键

| 实验 | 结果 | 我的判断 |
| --- | --- | --- |
| **v0.2-A：无 warm-up** | A_end **0.211**，B/C 接近随机 | ❌ 鸡生蛋问题仍然存在 |
| **v0.2-B：只给第一任务 warm-up** | A_end **0.455**，forgetting **0** | ✅ **目前最有希望** |
| **v0.2-C：每个任务都 warm-up** | A_init **0.340** → A_end **0.004** | ❌ 灾难性遗忘 |

---

# 最重要的发现

## ① Router warm-up 确实有效

v0.2-A：冷启动 Router 不知道选谁 → Node 得不到足够梯度 → Node 学不会 → Router 更不知道选谁，A 只有 **0.211**。

而 v0.2-B：第一任务 A → soft routing warm-up → A 建立能力 → 进入 hard routing → B/C，A 可以到 **0.455**，forgetting=**0**。

> **先建立一个"能力锚点"，再让动态路由接管，是可行的。**

## ② 但是"每个任务重新 soft warm-up"是错的

A: 0.340 → B warm-up → C warm-up → A: 0.004，forgetting=**0.845**。

> **Soft routing 本身不是解决方案。** 它让多个 Node 同时承担同一份梯度。

仓库自己已总结：**Full-soft per-task routing ≈ shared-projection overwrite**（多个模块共享梯度 → 模块隔离消失 → 后面的任务覆盖前面的能力）。

---

# 所以核心机制应该改成

不是 `Soft → Hard`，而是：
```
新任务 → Novelty Detector → Spawn/Reserve → Soft Assignment（仅新 Node）→ 能力形成 → Hard Routing → Module Lock
```

**关键：不是让整个网络 soft，而是只让"新任务对应的新模块"经历 soft warm-up。** 已经成熟的 Node（A Node/B Node/C Node）不要再被后续任务的 soft routing 大面积更新。

---

# Dynamic Neural Ecology

Node 不只是参数，它有：age / state / specialization / usage / memory / competence / plasticity。Router 不仅问"你适合这个 token 吗"，还问 **"你有没有资格学习这个 token？"**——这两个东西要分开。

Node 生命周期：NEW → WARMING → LEARNING → MATURE → SPECIALIZED → DORMANT → REACTIVATED，而非简单 alive/dead。

---

## 建议下一版：FDN-v0.3 Protected Expert Formation（只加三个机制）

- **A. 新 Node 专属 warm-up**：existing Node → hard/protected；new Node → soft
- **B. Competence Lock**：loss↓ + usage↑ + routing entropy↓ 达到 → Node→MATURE，成熟后降低 plasticity
- **C. Re-activation**：A 再出现，Router 找到 A Node 直接恢复（不重新 warm-up）

## 研究路线

```
v0 结构会乱 → v0.1 结构稳定 → v0.2 发现 Router cold-start → v0.2B 证明 first-task warm-up 可行 → v0.2C 发现 global soft routing 造成能力覆盖 → v0.3 新模块soft旧模块protected → ???
```

如果 **v0.3 能做到 A→B→C→A'，同时 A 能力基本不掉、B/C 形成独立 Node**，GPT 会认真认为：**我们不是在做一个 MoE 变体，而是在得到一种新的"动态模块形成机制"。**

`8a2b24a` 已把 12/12 tests 跑通，不再是代码工程问题，进入真正的机制实验阶段。**这次建议别急着上 BDH，先把 v0.3「新模块可塑、旧模块受保护」的机制打穿。**

---

*（正文完，web_extract 抓取。）*
