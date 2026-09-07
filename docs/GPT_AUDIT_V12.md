# GPT 审计 V12 —— de023ba（DCA training-free Interference Predictor 池）（2026-09-07）

> 来源：ChatGPT 分享「深夜问候聊天」（share t_6a9e3ce7…，第三方后端抓取全文）。
> 审计对象：`github.com/sinoclaw/fdn` commit `de023ba`（DCA training-free Interference Predictor 接入 DynPool）。
> 结论：**PARTIAL PASS，实验可信，无系统造假；但远不到「通用动态能力管理器」**。且**此线与 FusedFW→Transformer 主线不同优先级**，DCA 仅作辅助机制，**主线应继续走严格真实模型蒸馏 benchmark（FAIR_BENCH_V2）**。

---

## 一、de023ba 做了什么

把 DynPool 的「真 probe（深拷贝+训练→看旧能力掉多少，很贵）」换成**离线训练好的 MLP Interference Predictor**：输入 `[新技能 + 当前 capability state]` → 预测 damage → REUSE/SPAWN。运行时只前向预测。

## 二、版本轨迹（诚实性）

- **v1**：5 skill / 16 行 → train Sp=1.0、holdout Sp=-0.29 → predictor 错把 cross-family 复用 → extract acc≈0.002（灾难性遗忘）。commit 直接判 **NEGATIVE** 并指出 overfit（**好评，没藏失败**）。
- **v2**：换 ip3 平衡 48 行（每 family 多变体 + REUSE/SPAWN 平衡）→ holdout Sp=0.898 / consistency=0.812 / F1=0.857（超 0.7/0.8/0.8 判据）。但 DynPool 真实序列里 **transform_v0 → REUSE into math 是错的**（math+transform degradation 到 0.50-0.52）→ 整体判 **PARTIAL PASS**（**诚实**）。

## 三、四个验证设计问题（本次重点）

1. **train/test 不是完全独立的任务空间**：split 是「**每种 `(cap_family, new_family)` pair 内抽 30% 做 test**」，不是「整个 family-pair 留出」。→ 只能证明「**已见 family-pair 内、对没见过的 variant 有泛化**」，**不能**证明「对完全没见过的 capability/family combination 也泛化」。commit 说 "generalize on balanced data" 应改述为「在已见 family-pair 上对 held-out variants 有泛化」。
2. **threshold 有人为调优**：部署用固定 `self.thr=0.3`，但对账时 `for t in np.arange(0.0,0.6,0.01)` 搜索**最佳阈值**再报 consistency。→ `consistency=0.812` **不能**理解为「固定 thr=0.3 部署就有 81.2% 正确率」。应分开报 `fixed threshold=0.3` 与 `best threshold=X`。
3. **capability 训练协议变了**：PredPool 在 REUSE 时用的不是标准 `train_cap()`，而是 `2000 samples/Adam/lr=1e-3/30ep/batch=64`。**必须与构建 damage 数据时的真 probe 训练协议严格一致**，否则 predictor 学 damage_A 而 runtime 产生 damage_B → 部署漂移。标 **🟡 需继续审计**，非直接判错。
4. **"training-free" 措辞**：runtime 决策确免训练（MLP forward）；但 predictor **本身需离线生成 interference 数据 → probe training → 训练 predictor**。准确应叫 **"training-free at runtime"**，非「整个系统完全无需训练」。

## 四、「泛化」措辞过头

现在证明的是「已知 family-pair 内新 variant 可预测 interference」，**未证明**「未知 family / 未知 family-pair / 未知任务分布也可预测」。predictor 可能只学到 **family-level shortcut**（classify↔classify / extract↔extract / math↔math / transform↔transform）。一旦来真正新能力（code gen / vision / planning / tool use / long-context reasoning）可能直接失效 —— 而 DCA 真正想要的恰恰是后者。

## 五、最终评级

| 项目 | 判断 |
| --- | --- |
| v1 失败真实记录 | 🟢 |
| v2 真正改善 | 🟢 |
| holdout 存在 | 🟢 |
| runtime 避免真 probe | 🟢 |
| transform/math 边界解决 | 🔴 没解决 |
| "partial pass" 结论 | 🟢 很诚实 |
| family-pair 泛化 | 🟢 有证据 |
| unseen pair 泛化 | 🔴 没证明 |
| threshold 公平 | 🟡 有最佳阈值调参 |
| 明显造假 | 🟢 暂无 |

**评分**：实验态度 8/10，证据强度 5.5/10，结论措辞 7/10。

## 六、主线优先级（关键）

> 这条线和 FusedFW→Transformer 替代路线**不是同一优先级**。若最终目标 =「BDH/FusedFW 能否真替代 Transformer」，则 de023ba **不会让继续深挖**，最多作 DCA 辅助机制保存。
> **主线应继续往严格的真实模型蒸馏 benchmark 走（= FAIR_BENCH_V2）**。
