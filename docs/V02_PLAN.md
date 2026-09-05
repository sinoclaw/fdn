# FDN-v0.2 实验规划（GPT v0.1 复审 + 团队结论对齐）

> 依据：`docs/GPT_AUDIT_V01_REVIEW.md`（GPT 对 commit `cbafad0`=v0.1 的复审）+ 团队 EXPERIMENT_LOG 的 v0.1 判定。
> 目标：解决「动态路由鸡生蛋」，让 FDN 在**保留承重动态**下真的学会（而非 v0.1 把承重机制关掉后依旧学不会）。

---

## 0. 结论链（为什么这么规划）

| 版本 | 结构 | 承重动态 | acc_A | A/A' IoU | 结论 |
|---|---|---|---|---|---|
| v0 全动态 | 震荡（spawn42/merge42） | 开 | 0.055 | **1.0** | 「记住在哪里」成功，「怎么做」失败 |
| v0.1 稳定化 | 稳（merge0/15节点） | **关** | 0.045 | 0.667 | 结构治好，但承重机制被抽走 → 连 A 都学不会 |
| 单任务全动态 | — | 开（30ep无controller） | **0.598** | — | 铁证：memory/plasticity/τ/动态k 是「能学会」的承重 |

**统一结论**：
1. 结构震荡已从主因排除（v0.1 治好结构，acc 仍崩）。
2. 承重动态（memory/plasticity/τ/动态k）是「能学会」的前提（0.045 vs 0.598）。
3. **真正问题 = Dynamic Routing 鸡生蛋**：Router 冷启动不知谁擅长 A → 随机分配 → Node 信号不足 → 学不好 → Router 更不知所措。**这是优化问题，不是架构/结构问题。**

---

## 1. v0.2 核心方向（GPT 建议，团队认同）

```
                FDN
                 │
       ┌─────────┴─────────┐
       │                   │
 Dynamic Structure      Dynamic Dynamics     ← 保留承重动态（开 memory/plasticity/τ/动态k）
       │                   │
 freeze / prune       memory
 spawn cooldown       plasticity
 merge patience       dynamic τ
       │                   │
       └─────────┬─────────┘
                 ↓
          Dynamic Router   ← 新增：Router warm-up + soft→hard top-k curriculum
                 ↓
              Top-K
```

**一句话**：只把结构演化限制住（v0.1 已实现、保留），**把承重动态全部恢复**，并给路由加「软→硬」课程 + warm-up 来破鸡生蛋。

---

## 2. 两个递进实验（遵循「一次只动一组变量」闸门）

### v0.2-A：稳定结构 + 全动态（最小改动，先跑）
- 恢复全动态：`use_plasticity/memory/dynamic_tau/dynamic_k = True`
- 保留 v0.1 结构稳定化补丁（任务边界触发 Evolution、Spawn 冻结期、Merge 长期稳定）
- **唯一变量**：把「最小组合（静态权重）」→「全动态」，其余与 v0.1 相同
- 回答：稳定结构下，「保留承重动态」能否让 acc 从 0.045 回升？（验证「动态是承重」在稳定结构下成立）
- 预期：acc 应显著高于 v0.1；若仍低 → 鸡生蛋确认（路由本身挡住学习），进 B。

### v0.2-B：加 Router warm-up + soft→hard curriculum（针对鸡生蛋，主实验）
- 冷启动第 1 阶段：**soft routing**（全 Node 线性混合，gate=softmax over ALL nodes，温度先高后低），让每个 Node 都拿到梯度、Router 学到亲和
- warm-up 若干轮后：**退火到 hard top-k**（稀疏，只算选中 Node）
- 保留全动态 + 结构稳定化（同 A）
- 回答：soft→hard 课程能否破「鸡生蛋」，让 Router 稳定锁定任务亲和、Node 获得充足信号 → FDN 真的学会
- 判据见下。

---

## 3. 判据（先立判据再跑，机器原始输出为裁量）

**v0.2-A**
- Pass：`acc_A_init` 显著 > 0.045（如 ≥0.3）→ 承重动态在稳定结构下有效，鸡生蛋未显性阻断
- 若 `acc_A_init` 仍 <0.1 → 鸡生蛋成立，转 B

**v0.2-B**
- Pass（核心）：C 满足 ——
  (a) `acc_A_init ≥ 0.3`（A 真学会，非随机）；
  (b) `forgetting_A` 显著低于 v0（结构震荡版）且 A' 调用原有 Node（`node_overlap_A_A2 ≥ 0.7`）；
  (c) **A 保留曲线**：A→A 后 acc_A≥0.3；A→B→A 后 0.2；A→B→C→A' 后 ≥0.3 且用原 Node 组（调用非重学）。
- No-Go：三段均不达标 → 声明「当前 toy + 逐样本 top-k 路由模块化下，FDN 无法在稳定结构内学会（优化脆弱，非结构/机制缺失）」，如实交审计。

**曲线指标**（v0.2-B 核心可交付）：`A_retention_curve = [acc_A_after_AA, acc_A_after_ABA, acc_A_after_ABCA']`，配 `node_overlap` 逐段。

---

## 4. 运行序列（GPT 建议的递进曲线，替代直接 A→B→C→A'）

v0.2-B 用递进序列画出曲线：
```
A→A        （学 A，再测 A 保留）
A→B→A      （学 B 后测 A）
A→B→C→A'   （全序，测最终 retained + 调用）
```
> 比直接 A→B→C→A' 更能定位「学会后是否真的保留/调用」，也回应 GPT「这才是真正测 continual dynamic learning」。

---

## 5. 改动清单（落到代码）

| 文件 | 改动 |
|---|---|
| `model/fdn.py` | ① 加 `routing_temperature`、`soft_to_hard`（warmup步数、schedule）参数；② forward 里 `topk` → 按 schedule 用 full-softmax 或 top-k；③ 保梯度门控不变 |
| `growth/controller.py` | 已满足结构稳定化（本次不改或微调 spawn_cos_thr） |
| `experiments/continual.py` | 新增 `--profile v02`（A）与 `--profile v02b`（B，带 warm-up/curriculum + 递进序列）；扩展打分（A_retention_curve）。**v0.2-C（方案A）**：`--profile v02c` + `--warm_epochs N` + `--warm_mode first|every`，per-task warm-up（每任务前 N epoch 全班 soft） |
| `experiments/tasks.py` | 无改动（A2_add 已作为 A'） |
| `tests/` | 加：soft→hard 温度 schedule 正确性、warm-up 后 top-k 生效、retention 曲线采集 |

**运行命令（先 B 基线，再 A 对照）**：
```
.venv/bin/python experiments/continual.py --profile v02b --seed 0 --n_train 1200 --epochs_per_task 40 --run C --out results/summary_v02b.json
.venv/bin/python experiments/continual.py --profile v02  --seed 0 --n_train 1200 --epochs_per_task 40 --run C --out results/summary_v02a.json
```

---

## 6. 风险与缓解
- **soft→hard 若退火太快** → 鸡生蛋未破；缓解：warmup 轮数取大、温度指数退火、可跑敏感度。
- **toy 任务容量仍充足 / MoE 族本就难学**（v0 中 B/D/C 均远低于 A）→ 判据以「C 自身是否学会+保留」为主，不以超 B/D/A 为通过条件（把"容量饱和/任务太简单"作为已知局限单独声明，不混入判定）。
- **单 seed 噪声** → 关键判定跑 2-3 seed；本规划先 seed0 定方向，再扩 seed。

---

## 7. 待确认
- [x] 按 v0.2-A → v0.2-B 顺序推进（推荐，一次只动一组变量）
- [x] 还是直接上 v0.2-B（GPT 主建议，含 curry + warm-up + 递进曲线）
- [x] **v0.2-C（方案 A，per-task warm-up）**：v0.2-B 实测后 B/C 仍低（只 warm 首任务），Dad 拍板方案 A——每个任务前 warm_epochs 个 epoch 全班 soft，让每个能力组都能长出。

*（2026-09-06 起草，Dadv 拍板 v0.2-C。）*
