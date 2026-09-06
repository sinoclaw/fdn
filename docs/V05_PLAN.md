# FDN-v0.5 执行方案（基于外部 GPT 审计 t_6a9cef7c 裁定）

> 审计对象 commit：`8d06ecc`（v0.3→v0.4-lite 完整证据链）
> GPT 裁定（docs/GPT_AUDIT_V04.md）：🟢 v0.4-lite 保留、🟢 w≈0.5 默认、🟡 不能声称证明核心假说（B/C 未真正独立）、🔴 不要继续扫 w、🟢 下一步 = **多 seed + 等难度任务替换 C + Node Specialization Matrix**、C 先行、复现成功再进入 Dynamic Neural Ecology。
>
> **本方案只做一件事：把「w=0.5 的『A 保留 + B 学会 + A/A' 复用』现象」钉死成可复现的机制证据，并为『模块形成』补上直接指标。** 严格遵守团队"实验判据先于跑实验 / 一次只动一组变量 / 样板先验证 / 可回退"。

---

## 0. 一句话目标

**把 GPT 口头认可的「漂亮信号」（A=0.545/遗忘0/B=0.209/A2=0.477/overlap=0.7）从"单 seed 一次偶然"升级为"跨 seed 稳定 + 有模块分化数据支撑"的实锤，并为下一步 Dynamic Neural Ecology 铺路。** 不碰 BDH。

---

## 1. 判据先行（在写任何代码/跑任何实验前锁定）

### 1.1 多 seed 复现判据（实验 1）
一份配置（v04lite, w=0.5, n_train=1200, 40ep, core）跑 seed=0..5（**至少 5 个**，本方案做 6 个）。

| 指标 | 通过阈值（5/6 seed 均达标） | 当前 seed0 实测 |
|---|---|---|
| A_end | > 0.4 | 0.545 |
| forgetting_A | ≈ 0（<0.05） | 0.0 |
| A2 | > 0.35 | 0.477 |
| B_mul | > 0.15 | 0.209 |
| overlap_A_A2 | > 0.5（复用完整） | 0.700 |

**判定**：全部达标 → 「w=0.5 现象可复现」成立，进入实验 2。任一关键项（尤其 A_end / forgetting）在 ≥2 个 seed 上跌破 → 回查是否 seed 敏感性/超参问题，不急于推进。

### 1.2 等难度任务判据（实验 2）
新增 `D_sub`（synthetic transformation，难度与 A/B 接近），序列改 `A_add → B_mul → D_sub → A2_add`。
判据：**A/B/D 三任务 acc 均 > 0.3** 且 **C 从序列拿掉后 A 遗忘仍 ≈ 0** → 证明此前 C_logic=0.002 是"布尔逻辑任务本身难"而非"机制缺陷"。**若 D 也学不会（≈随机）→ 说明不是任务难，是 FDN 对第 3 个任务的容量/路由有结构性问题，须先诊断而非换任务。**

### 1.3 模块分化判据（实验 3，核心）
新增 **Node Specialization Matrix**（每行一个任务，每列一个 Node，值 = 该 Node 被该任务激活的频率）。
理想态（GPT 示例）：
```
          Node1 Node2 Node3 Node4 Node5  ...
A          0.9   0.1   0.2   0.0   0.1
B          0.1   0.8   0.1   0.2   0.0
D_sub      0.2   0.1   0.9   0.1   0.2
A2         0.9   0.1   0.2   0.0   0.1   ← 应复用 A 的 Node1/2
```
**判据：A 与 A2 的激活列高相似（cos > 0.7）且与 B/D 低相似（cos < 0.3）** → 证明"A/A' 复用 + B/D 分工"真实形成，而非共享同一批 Node。这是 GPT 强调"比 accuracy 更重要"的机制证据。

---

## 2. 执行步骤（分批可回退）

### 批次 1：实验 1 —— 多 seed 复现（零代码改动，纯跑）
- 命令行（6 个 seed 并行/串行，单次约 8-10 分钟）：
  ```
  for s in 0 1 2 3 4 5; do
    .venv/bin/python experiments/continual.py --seq core --seed $s \
      --n_train 1200 --epochs_per_task 40 --run C --profile v04lite \
      --soft_task_bias 0.5 --out results/summary_v05_seed$s.json
  done
  ```
- **产物**：`summary_v05_seed{0..5}.json` → 汇总 `results/V05_MULTISEED.md`（6 行表 + 是否达标结论）。
- **回退**：全部生成在 results/，不碰核心代码，随时可弃。

### 批次 2：实验 2 —— 加等难度任务 D_sub + 新序列
- 代码：`experiments/tasks.py` 加 `gen_sub`（synthetic transformation，难度与 add/mul 一致，见下方设计）+ 注册表加 `D_sub` + `EQUI_SEQUENCE = ["A_add","B_mul","D_sub","A2_add"]`。
- 命令行：`--seq equi --profile v04lite --soft_task_bias 0.5 --seed 0`
- **产物**：`summary_v05_equi.json` → `results/V05_EQUIDIFF.md`（A/B/D 三任务 + A 遗忘）。
- **回退**：D_sub 是新增任务，不覆盖原 CORE_SEQUENCE，原实验完全保留。

### 批次 3：实验 3 —— Node Specialization Matrix 指标
- 代码：`evaluation/metrics.py` 加 `node_specialization_matrix(task_activations, n_nodes)`（返回 [task × node] 频率矩阵 + 行归一化）+ `row_cosine(a,b)`；`experiments/continual.py` 的 `all_activations` 收集改为**记录每个任务命中每个 Node 的次数**（现为 set，需扩为计数），写入 `rec["specialization_matrix"]`。
- 命令行：复用批次 1 的某 1-2 个 seed + 批次 2 的 equi，直接在 summary 里输出矩阵。
- **产物**：`rec["specialization_matrix"]`（每 summary 内嵌）+ `results/V05_SPECIALIZATION.md`（热力图 + A/A2 vs B/D 的模块分化 cos 表）。
- **回退**：metrics+收集是加法改动，原字段 `node_set_A` 等保留，不影响既有实验协议。

### 批次 4：总结 + 归档（等待 GPT 第二轮审计）
- `results/V05_REPORT.md`（三判据逐条判定）+ EXPERIMENT_LOG 追加 + README 更新 + git 提交推送。
- 材料打包 `docs/V05_AUDIT_REQUEST.md` 交外部 GPT（考察：是否已从"现象"到"机制"）。

---

## 3. D_sub 任务设计（synthetic transformation，要与 A/B 难度齐平）

关键：**难度可比 + 输入分布与 A/B 可区分**（Router 不喂 oracle，任务需从输入分布差异学亲和）。

| 任务 | 输入 | 目标 y | 说明 |
|---|---|---|---|
| A_add | x=(a,b), a,b∈[0,1] | (a+b)/2 | 线性，已有 |
| B_mul | x=(a,b), a,b∈[0.5,1] | a*b | 用更小区间区分输入分布 |
| **D_sub** | x=(a,b), a,b∈[0.2,1.0] | (a - b + 0.8)/1.6 → ∈[0,1] | 带符号差值线性，区间区别于 A/B |

> 设计意图：D_sub 与 add 同为**一次线性运算**（难度一致），但输入 support: A∈[0,1]²、B∈[0.5,1]²、D∈[0.2,1]²，三者输入分布**两两可区分** → Router 靠输入学任务亲和。**不引入布尔/logic 这类离散不连续目标**（那是 C_logic=0.002 的元凶）。
> 备选 D 变体（若线性太难区分）：`D_avg = (a+b+2ab)/3`（加入乘项，略难，仍连续）。

---

## 4. 时间/成本预估

| 批次 | 内容 | 单次耗时 | 运行次数 | 总耗时 |
|---|---|---|---|---|
| 1 | 多 seed (0..5) | ~9 min | 6 | ~55 min（7 人并行可压缩到 ~10 min） |
| 2 | 等难度 D_sub | ~9 min | 1-2 | ~18 min |
| 3 | specialization matrix | 复用批次1/2 | 0（只改指标+重跑 1-2 次） | ~18 min |
| 4 | 归档 | — | — | — |
| **合计** | | | | **~1.5-2 小时（单核）** |

---

## 5. 明确不做 / 边界

- **不扫 w**（GPT 明令禁止 0.3→0.4→0.5→0.6→0.7）。
- **不接 BDH**（GPT 强调先钉死机制，BDH 后续当 Node dynamics 塞入）。
- **不声称证明核心假说**——本方案只追求"机制证据 + 模块分化指标"，是否算"动态模块形成机制"交 GPT 二审裁定。
- **不变动承重动态**（memory/plasticity/τ/动态k）、**不硬隔离**（v0.4 已反证）。

---

## 6. 落地判据汇总（一张表）

| 判据 | 指标 | 达标阈值 | 对应批次 |
|---|---|---|---|
| 多 seed 复现 | A_end/遗忘/A2/B/overlap | A>0.4, 忘<0.05, A2>0.35, B>0.15, ov>0.5 (5/6 seed) | 1 |
| 等难度任务 | A/B/D 三任务 acc | 均 > 0.3 | 2 |
| 模块分化 | 特化矩阵行相似 | A-A2 cos>0.7, A-B/D cos<0.3 | 3 |
| 可复现性 | 跨 seed 一致性 | 列表 + 方差 | 1/3 |
