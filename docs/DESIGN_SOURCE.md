# FDN-v0「全动态网络」设计源对话

> 由 ChatGPT 外部分享对话转 Markdown。这是**全新实验项目 FDN-v0（Fully Dynamic Network，全动态网络）**的设计源对话，完整记录其从零推演的过程（6 层动态、动态节点/路由/内存/参数演化、Evolution Controller、验收标准），可作为设计↔实现对齐的依据。

---

**👤 用户：**
哈喽

**🟢 ChatGPT：**
哈喽，老6 😎

这么晚还没睡？今晚准备折腾啥—— **BDH、蒸馏 DeepSeek、Agent Runtime，还是又准备搞个离谱的新项目？** 😂

**👤 用户：**
你是老6

**🟢 ChatGPT：**
哈哈，是我，老6本人 😎

你一喊“老6”，我就知道今晚又要开始整活了。

来吧，这次准备折腾什么？😂

**👤 用户：**
你现在是啥模型啊

**🟢 ChatGPT：**
我是 **GPT-5.6 Luna**。😎

就是你现在聊天的这个版本。

**👤 用户：**
可以免费用嘛

**🟢 ChatGPT：**
可以啊，老6 😎

**你现在用的 GPT-5.6 Luna，Free 免费版就能用。** OpenAI 官方目前明确写的是：Free 和 Go 用户默认使用 GPT-5.6 Luna，日常文字聊天基本不限量，但文件上传、图片生成、语音、数据分析等工具仍有各自的额度限制。[OpenAI Help Center](https://help.openai.com/en/articles/20001354-gpt-56-in-chatgpt/)

简单说：

- 🆓 **Free：GPT-5.6 Luna**

- 💰 **Plus/Pro：可以用更强的 GPT-5.6 Sol**

- 🧠 Free 用户的 **Think** 也由 Luna 提供支持

- 🔌 API 则是另外收费的，不等于 ChatGPT 免费聊天额度

所以你现在这个账号， **不用付钱也可以跟我聊**。😎 [OpenAI Help Center](https://help.openai.com/zh-hans-cn/articles/20001354)

Sources

**👤 用户：**
研究全动态模型啊

**🟢 ChatGPT：**
对， **研究全动态模型**。😎

而且我觉得咱们之前讨论的方向，可以再往前推一步：不是单纯搞“Liquid Neural Network”，而是搞一个真正意义上的 **Fully Dynamic Model（全动态模型）**。

传统模型大致是：

> **固定结构 + 固定参数 + 动态激活**

而我们想研究的是：

> **结构、参数、计算路径、记忆、时间尺度，全都可以动态变化。**

现在其实已经能看到一些相邻方向在往这个方向靠：比如 2026 年的 DYNcPNN 已经做到根据 concept drift 动态扩展网络；Dynamic Nested Hierarchies 更进一步，让层级能够添加、剪枝，并动态调整更新频率。[数字对象识别系统](https://doi.org/10.1007/s10618-026-01213-y)

但它们距离我们说的“ **全动态**”还差一截。

### 我建议我们把它定义成 6 层动态

```

                 ┌─────────────────────┐
                 │    Dynamic Model    │
                 └──────────┬──────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ↓                    ↓                    ↓
 Dynamic Structure     Dynamic Weights     Dynamic Routing
       │                    │                    │
       ↓                    ↓                    ↓
  增删/合并模块        参数实时演化          每token不同路径
       │                    │                    │
       └──────────────┬─────┴────────────────────┘
                      ↓
              Dynamic Memory
                      ↓
              Dynamic Time Scale
                      ↓
             Dynamic Compute
```

也就是说：

**1. 结构动态**

模型不是训练完就固定。

可以：

- 加 neuron

- 删 neuron

- 加 layer

- 合并 layer

- 拆分 module

- 创建新专家

- 删除没用的专家

这方面已经有研究基础。[Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1804338/full)

**2. 权重动态**

甚至不一定把：

```

W = constant
```

作为基本假设。

可以变成：

```

W(t+1) = W(t) + ΔW(x,t)
```

也就是 **权重本身是状态变量**。

Liquid 网络已经在做类似的连续动态，只不过目前大多集中在神经状态/时间常数，而不是整个大模型的全部参数。[科学直通车](https://www.sciencedirect.com/science/article/pii/S1877750325002340)

**3. 路径动态**

这个其实对 LLM 非常重要。

同一个模型：

```

token A
   ↓
Module 1 → Module 7 → Module 13

token B
   ↓
Module 2 → Module 4 → Module 18 → Module 21
```

不是传统 Transformer 那种：

```

所有 token
 ↓
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
...
```

而是 **每个 token 自己决定计算路径**。

* * *

### 4. Memory 动态

这才是我觉得最有意思的地方。

模型不再只有：

```

weights
```

而是：

```

Parameters
+
Working Memory
+
Long-term Memory
+
Episodic Memory
+
Structural Memory
```

而且这些东西都可以动态创建、修改、压缩、删除。

最近也出现了把长期记忆和多个短期记忆网络组合起来做 continual learning 的工作。[科学直通车](https://www.sciencedirect.com/science/article/pii/S0893608026002364)

* * *

### 5. 时间尺度动态

这个非常接近我们之前聊的 **BDH**。

不同状态可以拥有不同时间常数：

```

fast state      → token级
medium state    → sequence级
slow state      → task级
very slow state → knowledge级
```

甚至：

```

τ = f(x, state, surprise)
```

也就是说：

**模型自己决定“这个东西应该记多久”。**

Liquid Neural Network 的核心优势之一就是连续时间动态和输入调制的时间常数。[数字对象识别系统](https://doi.org/10.1016/j.aei.2025.104222)

* * *

### 6. Compute 动态

最后甚至：

```

简单问题 → 2步计算

普通问题 → 20步

困难问题 → 200步

发现自己不确定
      ↓
增加计算
      ↓
调用更多模块
      ↓
修改内部状态
      ↓
继续推理
```

所以模型不再是：

> **一次 forward pass**

而变成：

> **一个不断演化的计算系统。**

* * *

## 我甚至觉得可以给这个方向一个非常明确的公式

传统神经网络：

y=fθ​(x)

Liquid：

dtdh​=fθ​(h,x,t)

我们要研究的东西可以写成：

(St+1​,Wt+1​,Mt+1​,Ct+1​)=F(St​,Wt​,Mt​,Ct​,xt​)​

其中：

- S： **Structure**

- W： **Weights**

- M： **Memory**

- C： **Compute state**

甚至：

Modelt+1​=F(Modelt​,xt​,surpriset​,memoryt​)​

这就开始真正接近：

# **Self-Evolving Neural Network**

而不是简单的 Liquid Network。

* * *

而且老6，我觉得这次 **不要直接上大模型**。

咱们之前 BDH 那次踩坑，最大的问题就是一上来就想把东西塞进 LLM。

这次应该反过来：

**先做一个 10M～100M 参数的“全动态实验场”**。

先验证一个最核心的问题：

> **一个模型能不能在持续输入数据的过程中，自己改变自己的结构、参数、记忆和计算量，同时保持旧知识？**

这个一旦跑通，再考虑往 **BDH + Transformer/SSM + Dynamic Memory + Dynamic Routing** 上迁移。

而且我刚查了一下，2026 年已经有非常接近这个方向的工作，但目前基本都是 **动态结构 / continual learning / liquid dynamics 各自解决一部分**，还没有看到一个特别完整的“六维全动态 LLM”范式。[数字对象识别系统+2科学直通车](https://doi.org/10.1007/s10618-026-01213-y)

**所以这个方向，值得认真搞。** 😎

Sources

**👤 用户：**
怎么搞？

**🟢 ChatGPT：**
能搞，而且这次我建议 **别直接魔改 Transformer**。我们先做一个能跑、能测、能证明“全动态”成立的最小原型。

### 第一阶段：先定义“全动态”

咱们做一个 `FDN-v0`（Fully Dynamic Network）：

```

                 输入 x_t
                    │
                    ▼
             ┌─────────────┐
             │ Dynamic      │
             │ Router       │
             └──────┬──────┘
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Node A     Node B    Node C
          │         │         │
          └────┬────┴────┬────┘
               ▼         ▼
          Dynamic State / Memory
                    │
                    ▼
             Dynamic Output
```

关键不是“节点很多”，而是 **节点可以活着、死亡、复制、合并，参数还能自己变化**。

* * *

## ① 最底层：Dynamic Node

每个 Node 不再是固定神经元，而是一个小型动态单元：

ht+1​=ht​+Δt⋅fθ​(ht​,xt​)

同时增加一个可学习时间常数：

τi​=fϕ​(xt​,ht​)

于是每个节点都有自己的：

```

state
weights
time_constant
age
usage
importance
```

* * *

## ② Router：决定今天算谁

不要每次把所有 Node 都算一遍。

```
Python

nodes = router(x, state)

for node in nodes:
    state = node(state, x)
```

Router 输出：

```

Node 3
Node 7
Node 19
Node 31
```

下一次输入进来可能完全不同：

```

Node 2
Node 3
Node 11
Node 27
Node 42
```

这样才有真正的 **dynamic computation graph**。

* * *

## ③ Node Evolution：让结构自己变化

每隔 N 个 step 做一次结构评估。

例如：

```

usage < threshold
        ↓
      prune

error持续很高
        ↓
      split

两个节点高度相似
        ↓
      merge

某个模式经常出现
        ↓
      spawn
```

于是：

Gt+1​=E(Gt​,xt​)

这里的 G 就是模型结构。

**模型本身成为一个动态状态。**

* * *

## ④ Memory：不要把所有东西硬塞进权重

建立独立 Memory：

```

Short Memory
      ↓
Working Memory
      ↓
Long Memory
```

例如：

```
Python

memory.write(key, value)
memory.retrieve(query)
memory.consolidate()
memory.forget()
```

重点是：

**模型可以通过 Memory 学习，而不是每次都修改全部参数。**

* * *

## ⑤ Weight：真正做到动态参数

传统：

```
Python

y = layer(x)
```

我们改成：

```
Python

delta_w = plasticity(x, state)
w = w + delta_w
y = layer(x, w)
```

也就是：

Wt+1​=Wt​+ΔWt​

这样模型在 inference 阶段也可以发生有限的参数变化。

这一步要特别谨慎，因为 **灾难性遗忘** 会马上出现。

* * *

# ⑥ 最重要：Evolution Controller

把前面全部统一起来。

```

                  ┌──────────────┐
                  │ Evolution    │
                  │ Controller   │
                  └──────┬───────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   Structure          Weights           Memory
        ↓                ↓                ↓
   spawn/prune       plasticity      write/forget
        │                │                │
        └────────────────┼────────────────┘
                         ↓
                      Router
                         ↓
                    Computation
```

Controller 每隔一段时间判断：

```

loss
uncertainty
novelty
memory pressure
node usage
gradient statistics
compute budget
```

然后决定：

> **现在应该改变什么？**

* * *

# 最小实验我建议这样做

别碰 LLM。

直接：

```

PyTorch
+
100 个 Dynamic Nodes
+
Dynamic Router
+
External Memory
+
Evolution Controller
```

规模：

```

输入维度：128
hidden：256
Node：100
每次激活：5～15
参数量：约几十万～几百万
```

训练数据故意做成 **不断变化的任务**：

```

Task A
↓
Task B
↓
Task C
↓
Task A'
↓
Task D
↓
Task B'
```

例如：

```

阶段1：加法
阶段2：乘法
阶段3：排序
阶段4：加法变规则
阶段5：新的排序规则
```

然后观察：

### 固定网络

```

学习 A
 ↓
学习 B
 ↓
A 忘了
```

### 普通 Continual Learning

```

A → B → C
 ↓
通过 replay 尽量保持 A
```

### 我们的 FDN

目标：

```

A
 ↓
spawn Node group A

B
 ↓
spawn Node group B

C
 ↓
spawn Node group C

再次出现 A
 ↓
Router 找到 Node group A
```

也就是说：

**模型不是重新学习 A，而是“长出过 A”。**

这才是关键实验。

* * *

# 第二阶段才上 BDH

如果第一阶段成功，再把 Node 换成：

```

Dynamic Node
      ↓
BDH Cell
```

然后：

```

BDH
+
Dynamic Routing
+
Dynamic Memory
+
Structural Evolution
+
Adaptive Time Constant
```

最终形成：

FDN=Dynamic Structure+Dynamic Weights+Dynamic Routing+Dynamic Memory+Dynamic Time+Dynamic Compute​

* * *

## 最后才考虑 LLM

那时候不是：

> “怎么把 Transformer 改成动态？”

而是：

> **“怎么把语言能力接入已经证明有效的动态神经系统？”**

可以让 Transformer / SSM / BDH 变成不同类型的 **Dynamic Cells**：

```

                 FDN Runtime
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
     BDH Cell     SSM Cell    Attention Cell
        │            │            │
        └────────────┼────────────┘
                     ↓
              Dynamic Memory
```

这时候就很有意思了。

**模型不再是一坨固定权重，而更像一个会生长、会退化、会记忆、会重组的计算生态系统。**

如果你要让 **Hermes 直接开干**，我下一步可以直接给你出一份 **FDN-v0 的完整技术规格 + PyTorch 项目目录 + 核心代码接口 + 第一轮实验指标**，让它照着实现，而不是再像上次 BDH 那样瞎折腾。

**👤 用户：**
好的

**🟢 ChatGPT：**
好，老6。😎

这次咱们就按 **“先跑通、再证明、最后扩展”** 来，别一上来挑战百万/十亿参数。

我建议第一版项目直接定名：

# FDN-v0：Fully Dynamic Network

核心目标只有一句话：

> **模型在持续学习过程中，可以自主改变计算结构、参数状态、记忆和计算量，同时尽可能保持旧知识。**

第一版先做 **100 万参数以内**，重点验证机制，不追求模型能力。

如果你准备让 Hermes 开工，我可以下一条直接给你一份 **可以原封不动扔给 Hermes 的开发任务书**，包括：

- 项目目录

- PyTorch 接口

- Dynamic Node

- Dynamic Router

- Spawn / Prune / Merge

- Dynamic Memory

- Plastic Weight

- Evolution Controller

- 训练数据生成器

- baseline

- 实验指标

- ablation 实验

- checkpoint / 日志

- 第一版验收标准

这样它就可以直接开始写代码。

**👤 用户：**
好的

**🟢 ChatGPT：**
那就开干，老6。😎

我给你定一个 **不容易再次把一天时间折进去的 v0 方案**：第一版禁止碰 LLM、禁止碰 BDH、禁止搞复杂 CUDA 优化，先证明“全动态”这件事本身能工作。

### FDN-v0 验收标准

必须同时出现这 6 个能力：

1. **结构动态**：Node 能 `spawn / prune / merge`

2. **参数动态**：推理过程中允许局部 `ΔW`

3. **路由动态**：不同输入激活不同 Node

4. **记忆动态**：能写入、读取、巩固、遗忘

5. **时间动态**：不同 Node 有不同 `τ`

6. **计算动态**：简单输入少算，困难输入多算

最重要的实验：

```

Task A → B → C → A'
```

要求：

```

学习 A
↓
学习 B
↓
学习 C
↓
A' 出现

不是重新训练 A
而是重新激活/调用原来的结构
```

同时记录：

```

accuracy
forgetting
active_nodes
total_nodes
spawn_count
prune_count
merge_count
memory_size
FLOPs / sample
inference_steps
parameter_drift
```

### 第一阶段只做这张图

```

              Input
                │
                ▼
        ┌──────────────┐
        │ Dynamic      │
        │ Router       │
        └──────┬───────┘
               │
        Top-K Nodes
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
   Node       Node      Node
     │         │         │
     └─────────┼─────────┘
               ▼
        Working Memory
               │
               ▼
            Output
               │
               ▼
      Evolution Controller
        │      │      │
      Spawn  Prune  Merge
```

**先把这个闭环跑通。**

等 Hermes 写出来第一版，你把 **代码、报错或者实验结果直接甩给我**。我可以帮你逐层检查，尤其是防止它偷偷把“动态模型”实现成一个 **固定模型 + 几个 if 判断** 😂。

下一步就可以继续把它往 **BDH + FDN + LLM** 推。
