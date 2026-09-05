# FDN-v0 实验日志（EXPERIMENT_LOG）

> 记录 FDN-v0 从设计到实现再到实验的全过程。给外部审计（GPT）对齐「设计 → 实现 → 结果」三层。

## 2026-09-05 项目启动

### 源起
- 用户给出一份 ChatGPT 深度设计对话（存 `docs/DESIGN_SOURCE.md`），提出 **FDN-v0（Fully Dynamic Network）**：一个能自主改变结构/参数/路由/记忆/时间尺度/计算量的全动态网络。
- 明确要求：与 GSLM 项目同款流程——建公开 GitHub 仓库、写码、跑实验、推结果、交外部 GPT 审。

### 文献查重（先查重定位新颖点）
2026 年已有高度重叠工作：
| 来源 | 覆盖的 FDN 机制 |
|---|---|
| MoLEM（arXiv:2605.21951） | 动态 MoE + 潜记忆注入 + 冻结基座 |
| Dynamic Nested Hierarchies（Frontiers 2026） | 层级增删 + 惊奇信号调频（终身学习自进化） |
| Modular Continual Learning（arXiv:2604.14375） | 模块化路由 + 冻结专家 + 0 干扰 |
| IBF（arXiv:2604.07108） | 连续学习动力学理论 |

**结论**：FDN-v0 本体不是全新架构。相对新颖点 = 把 6 维动态收进**一个小型化路由模块化网络**，并整体验证「长出能力组 → 再调用而非重学」这一机制。该点正是 GSLM No-Go 复盘指出的**真模块化缺口**（GSLM 增长节点无独立计算路径、能力落在共享投影）。故 FDN-v0 值得做最小实证（不投 GPU，CPU 可复现）。

### 设计蓝图
见 `docs/DESIGN.md`（假说、架构图、A/B/C/D 对照矩阵、任务集、指标、判定准则）。

### 实现要点（对照 GSLM 缺口的关键修复）
- **每个 Node = 完整独立子网**（in_proj + 循环单元 + 独立 out_head + 独立 router key），被 Router 选中才参与计算 → 真正独立计算路径，而非共享投影里的身份。
- **Router**：查询向量 `q=W_q·x` 与每 Node key `k_i` 点积做 top-k 路由；gate 是软 softmax，**门控保留梯度**（否则 Router 学不到任务亲和）。
- **Evolution Controller**：spawn（新任务分布 → 继承+分化长出新 Node）/ prune（低用量归档）/ merge（key 高余弦去重）；生长用 `optimizer.add_param_group` 保留旧优化器状态。
- **记忆**：外部 KV Memory，训练时 write，供任务再现检测 + 辅助输出。
- **可塑权重**：推理期有界 Hebbian `ΔW`（衰减+裁剪）。
- **时间**：每 Node `τ_i=sigmoid(U_tau[x;h])`。
- **计算**：路由门控熵越高激活越多 Node（k 动态）。

### 实现过程踩坑（已修复）
1. `self.h` buffer 的**视图复用**导致 autograd inplace 版本错 → 读状态时 `detach().clone()`。
2. `torch.stack(out_list).unsqueeze(1)` 多包一层 → 与 mem_ret 相加触发广播维度错 → 去掉 `unsqueeze`。
3. 基线 `float(gate)` 丢掉 Router 梯度，且 `outs[b]=contrib` 丢梯度 → 改为与 FDN 相同的「含梯度张量 + stack」，保证 A/B/D/C 对比公平。
4. 单任务诊断暴露**状态未清零**导致回归污染 → toy 独立回归任务改为每次前向 `reset=True`；并补上训练时 `write_memory`。

### 单元测试
`tests/test_fdn.py`：9/9 通过（结构 spawn 真增长、prune/merge 改 active 集合、plastic 有界、路由随输入不同、高熵输入 k 更大、τ 不同、指标正确）。

### 运行
- 环境：`/data/fdn/.venv`（torch 2.14 CPU，清华源）
- 命令：`.venv/bin/python experiments/continual.py --seq core --seed 0 --n_train 1200 --epochs_per_task 40 --run all --out results/summary_v0.json`
- 顺序任务 A_add→B_mul→C_logic→A2_add，40 epoch/任务，对比 A/B/D/C。

（实验完成后在本文件尾部追加结果与判定。）
