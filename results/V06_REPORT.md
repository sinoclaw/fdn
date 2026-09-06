# FDN-v0.6 实验结果报告（V06_REPORT，如实版）

> 依据 GPT 二审判定（docs/GPT_AUDIT_V05.md）：v0.6 = Node autonomous specialization（Novelty-triggered Spawn + Node Competence + Plasticity Decay）。
> **三件事**：① Node 级 Novelty-Spawn 判据启动（✓）；② Plasticity Decay 5 态（✓）；③ 任务边界 freeze（救 A/B，但 40ep 下 A 学不会是鸡生蛋）。
> **最简实验**：`A→B→A'`（GPT 指定），6 seed，freeze_old，40ep。
> **核心结论：v0.6 的机制都在动（spawn 触发/5 态运作/freeze 生效），但「模块分化」仍未实现（cos(A,B) 全 >0.99、disjoint 全 0、A 遗忘 6 seed 里 4 个 >0.8）——不能声称 Node autonomous specialization 达成。** 这是一个有价值的反证。

---

## 0. 一句话结论

**v0.6 三件机制（Node 级 spawn / Plasticity 5 态 / freeze）均成功"启动"，但没有一个让"任务分化到不同 Node 组"发生。** 6 seed 的 specialization cos(A,B) 全 >0.99、disjoint 全 0、A/B 共用同一批 node。**"Router 发现已有能力；Node Ecology 产生新能力"的分工在 v0.6 未兑现，核心指标未达标。**

---

## 1. 批次 1：Node 级 competence/novelty 状态（纯测量，✓）

- +`node_novelty(1-cos)` / `node_error` buffer，forward 选中 node 时更新。
- **修 competence 信号源 bug**：原 `usage_share×(1-全局熵)` 在未选中 node 上恒 0（competence 死掉）→ 改 `归一化 error 倒数`（error 大 = competence 低 = gap 高）→ 正确反映 node 胜任度（node8 error=109→comp=0.36；node0/5/6 error<2.7→comp=0.87-0.89）。
- **修正陷阱**：未使用 node（error=0）被误判为"金牌"，置中性 0.5。

## 2. 批次 2：Node 级 Novelty-Spawn（✓ 触发，✗ 分化）

- spawn 改成 `score = 0.4·norm(novelty) + 0.3·norm(error) + 0.3·(1-norm(comp))`，score>0.5 触发（**关键纠偏**：硬 AND"所有 used node comp<0.5"永不触发，因为 A 学会的 node 拉高均值）。
- **探针验证 spawn 首次在真实训练触发**：spawn=2，`last_spawn_reason={score:0.512, avg_novelty:0.996, avg_error:0.945, sustained:True, task:3}`——B/A' 边界长出 node12。
- **但 node12 被 B 和 A' 共用**（不分 B 专属）——分化未形成。

## 3. 批次 3：Plasticity Decay 5 态（✓） + freeze_old（✓ 救 A/B 但鸡生蛋）

### 3.1 Plasticity 5 态
- +`life_stage`/`last_active_epoch` buffer；update_lifecycle 由"单一 mature×0.1"改为按 life_stage 分级衰减（NEW 0.98/WARMING 0.95/LEARNING 0.90/MATURE 0.85/DORMANT 0.98/REACTIVATED 1.15）。
- 探针 5 态运作正常（life_stage=[3,3,4,4,...] MATURE/DORMANT/NEW 分布合理）。

### 3.2 freeze_old（关键发现 + 假突破）
- **本质缺口**：Plasticity Decay 只保护 Hebbian 推理期，**不拦 Adam 主训练梯度**（A 的 node 在 B 仍被选中 → Adam 覆盖 → A 遗忘）。
- **freeze_old 解法**：任务边界冻结成熟旧 node（`requires_grad=False`，Adam 跳过）。
- **短跑（24ep）"假突破"**：forgetting_A 0.992→0.0、A_end 0.002→0.449、B 0→0.904！
- **但完整 40ep 暴露真相（诚实反证）**：A_curve 起点 **0.021**（A 从一开始就没学会）——freeze 救不了"没学会"，**A 任务本身的 Router 冷启动鸡生蛋仍在**。

---

## 4. 批次 4：多 seed 结果（核心反证）

`A→B→A'`，6 seed，profile v06 + freeze_old，40ep：

| seed | A_end | forgetting | B_mul | A2 | cos(A,A') | **cos(A,B)** | disjoint |
|---|---|---|---|---|---|---|---|
| 0 | 0.021 | 0.847 | 0.0 | 0.020 | 0.999 | **0.997** | 0 |
| 1 | 0.699 | 0.0 | 0.0 | 0.822 | 1.000 | **0.999** | 0 |
| 2 | 0.002 | 0.992 | 0.0 | 0.000 | 1.000 | **1.000** | 0 |
| 3 | 0.016 | 0.857 | 0.713 | 0.008 | 1.000 | **1.000** | 0 |
| 4 | 0.051 | 0.469 | 0.090 | 0.037 | 0.998 | **0.992** | 0 |
| 5 | 0.002 | 0.964 | 0.0 | 0.002 | 0.999 | **0.988** | 0 |

### 达标统计（GPT 判据）
| 判据 | 阈值 | 达标 |
|---|---|---|
| A_end | >0.4 | 1/6 |
| B_mul | >0.3 | 1/6 |
| forgetting | <0.05 | 1/6 |
| cos(A,A') | >0.7 | 6/6（但全近 1，非真分化） |
| **cos(A,B)** | <0.5 | **0/6** ❌ |
| **disjoint** | >0 | **0/6** ❌ |

---

## 5. 判定（诚实，不渲染）

- ✅ **机制都在动**：spawn 触发（2）、Plasticity 5 态运作、freeze 生效（freeze_count 正常）——v0.6 的三件机制**成功启动**。
- ❌ **模块分化未实现**（核心失败）：cos(A,B) 全 >0.99、disjoint 全 0——A/B 仍共用同一批 node。**"Node 自主专化"未兑现。**
- ❌ **A 仍易遗忘**：6 seed 里 4 个 forgetting>0.4（seed0/2/3/5），仅 seed1 偶发 A=0.699。
- ❌ **鸡生蛋未破**：A_curve 起点普遍 0.02（A 任务本身没学会——Router 冷启动问题，v0.2 的 warm-up 未在 v0.6 保留）。

## 6. 意义

1. **v0.6 的方向不能宣布成功**——GPT 的"Node 自主专化"在 FDN 当前架构下**没有产生 specialization pressure**（这与 v0.5 结论一致：任务间无模块分化仍是根本缺口）。
2. **freeze_old 是承重防遗忘机制**（40ep 对比 24ep 证实：A 遗忘主要来自"没学会"而非"被覆盖"），值得保留但非充分。
3. **A 任务学不会是更深层的鸡生蛋**——A 本身没学好（A_curve 起点 0.02），后续所有机制在"无米之炊"上打转。**下一步应回到"先让 A 学好"（warm-up / curriculum）**，而不是继续加分化机制。
4. **cos(A,A') 全 1 是"虚高分"**——所有任务共用 node，任何两任务行其实都近相同，cos 无区分度。

---

## 7. 原始数据

- `results/summary_v06_seed{0-5}.json`（6 seed）
- 本报告 `results/V06_REPORT.md`
- `results/EXPERIMENT_LOG.md`（批次 1/2/3/3+ 日志）
