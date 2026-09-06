"""experiments/continual.py — FDN-v0 核心实验：顺序学习 A→B→C→A'，测量遗忘/Node复用/结构动作。

运行：
  cd /data/fdn && .venv/bin/python experiments/continual.py --seq core --out results/summary_v0.json
"""
import argparse, json, time, os
import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import tasks as T
from experiments.baselines import StaticMLP, StaticMoE
from model.fdn import FDN
from growth.controller import EvolutionController
from growth.plasticity import plastic_magnitude
from evaluation import metrics as M


def make_batches(x, y, bs, shuffle=True, seed=0):
    n = x.shape[0]
    rng = np.random.RandomState(seed) if shuffle else None
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        idx = order[i:i+bs]
        if len(idx) == 0:
            continue
        yield torch.from_numpy(x[idx]), torch.from_numpy(y[idx])


def predict(model, x, dynamic=False):
    model.eval()
    with torch.no_grad():
        res = model.forward(x, reset=True)
        if isinstance(res, tuple):
            return res[0].numpy().reshape(-1), res[1]
        return res.numpy().reshape(-1), {}


def evaluate_model(model, seq_name, seed, task_ids=None, counts=False):
    """对每个任务生成评估集并测准确率 + 记录激活 Node。
    v0.4：若启用 task 约束，评估每任务前 set_task(任务 id)，保证用其专属 Node 组评估。
    counts=True：返回 {task: {node_id: count}}（Node Specialization Matrix 用）。
    """
    accs = {}
    activations = {}
    ncounts = {}
    for i, tname in enumerate(seq_name):
        x, y = T.TASKS[tname](512, seed)
        # v0.4：按当前任务 id 评估（若 task_ids 提供 pos 映射）
        if hasattr(model, "set_task") and task_ids is not None:
            model.set_task(task_ids[i])
        pred, info = predict(model, torch.from_numpy(x))
        accs[tname] = M.accuracy(pred, y, T.TOL)
        if "idx" in info:
            arr = info["idx"].reshape(-1).tolist()
            activations[tname] = sorted(set(arr))
            if counts:
                cnt = {}
                for ni in arr:
                    cnt[ni] = cnt.get(ni, 0) + 1
                ncounts[tname] = cnt
    return accs, activations, ncounts


def ensure_task_nodes(model, task_id, cfg, seed, optimizer=None):
    """v0.4：保证任务 task_id 有专属 Node 组（强制不相交）。
    任务 0：把初始未归属 Node（task_owner=-1）全部划归 task0（A 用足初始容量）。
    后续任务：spawn init_per_task 个专属 Node（owner=当前任务）。
    动态参数：optimizer 在 run_one 开头只含初始参数，spawn 后新 Node 参数须 add_param_group（保留旧 state）。
    """
    if not hasattr(model, "task_owner") or model.task_owner.numel() == 0:
        return
    existing = int((model.task_owner == task_id).sum().item())
    want = cfg.get("init_per_task", cfg["init_nodes"])
    if existing >= want:
        return
    # 任务 0：把未归属 Node（-1）划给 task0，补足到 want（初始 capacity 留给第一个任务）
    if task_id == 0:
        unowned = (model.task_owner == -1).nonzero().reshape(-1).tolist()
        for ni in unowned:
            model.task_owner[ni] = 0
            model.n_owned += 1
    # 补足专属 Node（spawn 时 model.current_task 已是 task_id，_append_node 会正确归属）
    existing = int((model.task_owner == task_id).sum().item())
    need = max(0, want - existing)
    for _ in range(need):
        model._append_node(parent_idx=None, noise=0.0)
        if hasattr(model, "_grow_keys"):
            model._grow_keys()
    # 动态参数：新 Node 参数加入优化器（保留旧 param_group state）
    if optimizer is not None and hasattr(model, "node_params_for_optim"):
        for ni in range(model.node_count()):
            if int(model.task_owner[ni]) == task_id:
                params = model.node_params_for_optim(ni)
                existing_params = {id(p) for g in optimizer.param_groups for p in g["params"]}
                to_add = [p for p in params if id(p) not in existing_params]
                if to_add:
                    optimizer.add_param_group({"params": to_add})


def train_task(model, x, y, optimizer, epochs, bs, dynamic, use_lifecycle=False):
    model.train()
    lossf = nn.MSELoss()
    for ep in range(epochs):
        # v0.3：per-node 保护在 model.forward 内部生效（未成熟 Node warm、成熟 Node protected），
        # 不再做"全班 soft"（v0.2-C 教训）。此处仅收集 loss/entropy 作为 Competence Lock 信号。
        task_loss_sum = 0.0
        nbatch = 0
        last_entropy = 0.0
        for xb, yb in make_batches(x, y, bs, shuffle=True):
            optimizer.zero_grad()
            res = model.forward(xb, reset=True)       # toy 独立回归：每次前向清零状态
            out = res[0] if isinstance(res, tuple) else res
            loss = lossf(out.reshape(-1), yb)
            loss.backward()
            optimizer.step()
            task_loss_sum += float(loss.item()); nbatch += 1
            if isinstance(res, tuple) and isinstance(res[1], dict):
                last_entropy = res[1].get("entropy", 0.0)
            # FDN：训练时把 (输入, 目标) 写入外部记忆（use_memory 内部关闭则 noop）
            if dynamic and hasattr(model, "write_memory"):
                model.write_memory(xb, yb)
            # 结构演化改为「任务边界触发」，见 run_one 的 controller.task_boundary
        # —— v0.3 Competence Lock：每 epoch 结束按 loss/entropy/usage 更新每 Node maturity ——
        if use_lifecycle and hasattr(model, "update_lifecycle"):
            avg_loss = task_loss_sum / max(1, nbatch)
            # v0.6：传 node_error（真实预测误差）驱动 competence（GPT：competence gap 是本 batch 内 node 误差信号）
            node_err = getattr(model, "node_error", None)
            model.update_lifecycle(avg_loss, last_entropy, node_err=node_err)
            model.tick_epoch()


def run_one(name, seq, cfg, seed):
    """跑一个模型。name ∈ {A,B,D,C}。返回 summary 子字典。"""
    rec = {"model": name, "seed": seed, "param_count": None}
    if name == "A":
        model = StaticMLP(dim=T.DIM, hidden=cfg["mlp_hidden"])
        dynamic, controller = False, None
    elif name == "B":
        model = StaticMoE(dim=T.DIM, hidden=cfg["hidden"], r=cfg["r"], n_nodes=cfg["init_nodes"], k=cfg["k_fixed"])
        dynamic, controller = False, None
    elif name == "D":
        model = StaticMoE(dim=T.DIM, hidden=cfg["hidden"], r=cfg["r"], n_nodes=cfg["eq_nodes"], k=cfg["k_fixed"])
        dynamic, controller = False, None
    else:  # C = FDN-v0
        model = FDN(dim=T.DIM, hidden=cfg["hidden"], r=cfg["r"],
                    initial_nodes=cfg["init_nodes"], base_k=cfg["k_fixed"],
                    kmin=cfg["kmin"], kmax=cfg["kmax"],
                    memory_max=cfg["memory_max"],
                    plastic_lr=cfg["plastic_lr"], plastic_cap=cfg["plastic_cap"],
                    use_plasticity=cfg["use_plasticity"], use_memory=cfg["use_memory"],
                    dynamic_tau=cfg["dynamic_tau"], dynamic_k=cfg["dynamic_k"],
                    warmup_steps=cfg.get("warmup_steps", 0), warm_temp=cfg.get("warm_temp", 3.0),
                    task_constraint=cfg.get("task_constraint", False),
                    n_tasks=cfg.get("n_tasks", 4), soft_task_bias=cfg.get("soft_task_bias", 0.0))
        dynamic, controller = True, EvolutionController(
            spawn_cos_thr=cfg["spawn_cos_thr"], prune_usage_thr=cfg["prune_usage_thr"],
            merge_cos_thr=cfg["merge_cos_thr"], young_tasks=cfg["young_tasks"],
            freeze_tasks=cfg["freeze_tasks"], merge_patience=cfg["merge_patience"],
            merge_usage_thr=cfg["merge_usage_thr"],
            use_node_spawn=cfg.get("use_node_spawn", False))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    rec["param_count"] = model.param_count()

    timeline = []          # 每任务阶段后的全任务准确率
    all_activations = {}   # task -> set(node)
    per_task_best = {}
    final_act_counts = {}  # 最后一次评估的 Node 计数（specialization matrix 用）

    for pos, task in enumerate(seq):
        x, y = T.TASKS[task](cfg["n_train"], seed + pos * 977)
        # v0.4：任务亲和约束 —— 任务边界注入当前任务 id，并为每个任务 spawn 专属 Node 组（强制不相交）
        #   v04lite（软亲和）只 set_task 加偏移，不 spawn 专属 Node（保持 v0.3 共享结构）
        if dynamic and hasattr(model, "set_task"):
            model.set_task(pos)          # 用任务内部序号 pos 作为 task_id（A=0,B=1,C=2,A'=3）
            if cfg.get("task_constraint", False):
                ensure_task_nodes(model, pos, cfg, seed, optimizer=opt)
        # v0.3：per-node protected routing（在 forward 内生效）+ Competence Lock（use_lifecycle）
        train_task(model, x, y, opt, cfg["epochs_per_task"], cfg["bs"], dynamic,
                   use_lifecycle=cfg.get("use_lifecycle", False))
        # C. Re-activation：进入新任务前，先看哪些成熟 Node 与新任务亲和（复用而非重新 warm）
        rec.setdefault("reactivation", {})[task] = []
        if dynamic and hasattr(model, "reactivate_score"):
            qq = model.q(torch.from_numpy(x[:8]).float()).detach()
            rec["reactivation"][task] = model.reactivate_score(qq.mean(dim=0))
        if dynamic and controller is not None:
            controller.task_boundary(model, opt, x)   # 结构演化：任务边界触发（降频）
        accs, act, act_counts = evaluate_model(model, seq, seed + 1000 + pos,
                                   task_ids=list(range(len(seq))), counts=True)
        timeline.append({"after": task, "acc": accs})
        all_activations[task] = set(act.get(task, []))
        final_act_counts = act_counts   # 记住最后一次评估的 Node 计数（specialization matrix 用）

    # ---------- 关键指标 ----------
    def acc_after(task):
        return timeline[-1]["acc"].get(task, None)

    def acc_initial(task):
        idx = timeline[0]["acc"].get(task, None)
        return idx

    # 遗忘（对 A：finish/开始）
    a_init = timeline[0]["acc"]["A_add"]
    a_end = timeline[-1]["acc"]["A_add"]
    rec["forgetting_A"] = M.forgetting(a_init, a_end)
    rec["acc_A_init"] = a_init
    rec["acc_A_end"] = a_end

    # A 与 A'（=A2_add）Node 激活重合（IoU）
    setA = all_activations.get("A_add", set())
    setA2 = all_activations.get("A2_add", set())
    rec["node_overlap_A_A2"] = M.set_iou(setA, setA2)
    rec["node_set_A"] = sorted(setA)
    rec["node_set_A2"] = sorted(setA2)

    # A 保留曲线（GPT 强调）：每阶段训练后 A_add 精度（若序列以 A 或 A' 结尾则有意义）
    rec["A_curve_after_each"] = [tl["acc"].get("A_add", None) for tl in timeline]

    # Node→任务亲和（是否用不相交 Node 组）
    subs, disjoint = M.node_task_affinity(
        {t: all_activations[t] for t in seq if all_activations.get(t)}, model.node_count() if dynamic else len(getattr(model, "nodes", [])))
    rec["task_affinity_disjoint_frac"] = disjoint
    rec["task_node_groups"] = {t: sorted(list(v)) for t, v in subs.items()}

    # FDN-v0.5：Node Specialization Matrix（GPT 强调，比 accuracy 更重要，直接测「模块分化」）
    n_nodes = model.node_count() if dynamic else len(getattr(model, "nodes", []))
    try:
        task_counts = {t: [final_act_counts.get(t, {}).get(i, 0) for i in range(n_nodes)] for t in seq}
    except Exception:
        task_counts = {}
    if task_counts:
        mat, mtx_tasks = M.node_specialization_matrix(task_counts, n_nodes)
        rec["specialization_matrix"] = mat
        rec["specialization_tasks"] = mtx_tasks
        rec["specialization_cos"] = {f"{a}~{b}": round(c, 3) for (a, b), c in M.row_cosine_pairs(mat, mtx_tasks).items()}

    if dynamic:
        rec["final_nodes"] = model.node_count()
        rec["spawn_count"] = controller.spawn_count
        # v0.6：记录最近一次 spawn 的触发信号（novelty/error/competence gap）——诊断是否 Node 级判据在起作用
        if getattr(controller, "last_spawn_reason", None) is not None:
            rec["last_spawn_reason"] = controller.last_spawn_reason
        rec["prune_count"] = controller.prune_count
        rec["merge_count"] = controller.merge_count
        rec["memory_size"] = model.memory.size
        rec["plastic_magnitude"] = plastic_magnitude(model)
        rec["k_used"] = cfg["k_fixed"]
        # v0.3：Node 生命周期分布（maturity/competence）
        if hasattr(model, "maturity") and model.maturity.numel() > 0:
            rec["maturity"] = [round(float(v), 3) for v in model.maturity.detach().cpu().tolist()]
            rec["competence"] = [round(float(v), 3) for v in model.competence.detach().cpu().tolist()]
            rec["n_mature"] = int((model.maturity >= model.mature_thr).sum().item())
        rec["usage"] = [float(v) for v in model.usage.detach().cpu().tolist()]
    else:
        rec["final_nodes"] = len(model.nodes) if hasattr(model, "nodes") else 1

    # v0.6：Node 级 telemetry（novelty/error/competence/maturity）——诊断「现有 node 对 B 是否 competence 低」
    if dynamic and hasattr(model, "node_telemetry"):
        rec["node_telemetry"] = model.node_telemetry()

    rec["timeline"] = timeline
    rec["final_acc"] = timeline[-1]["acc"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="core", choices=["core", "long", "retention", "equi", "mini"],
                    help="retention：跑 A→A' / A→B→A' / A→B→C→A' 三组，测 A 保留曲线；equi：用等难度 D_sub 替换 C_logic，测 A/B/D 三任务；mini：A→B→A'（v0.6 最简）")
    ap.add_argument("--out", default="results/summary_v0.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--epochs_per_task", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--run", default="all", choices=["all", "A", "B", "D", "C"])
    ap.add_argument("--profile", default="v0", choices=["v0", "v01", "v02", "v02b", "v02c", "v03", "v04", "v04lite", "v06"],
                    help="v0=全动态；v01=StabilityPatch；v02=稳定结构+全动态；v02b=首任务warm；v02c=每任务warm；v03=Protected Expert Formation；v04=+硬隔离(已反证)；v04lite=+软任务亲和(soft task bias)；v06=Node autonomous specialization(Novelty-Spawn)")
    ap.add_argument("--init_per_task", type=int, default=4,
                    help="v04：每个任务 spawn 的专属 Node 数（强制不相交路由）")
    ap.add_argument("--soft_task_bias", type=float, default=0.3,
                    help="v04lite：软任务亲和权重 w（q += w*task_emb[task]，不硬屏蔽）")
    ap.add_argument("--spawn_patience", type=int, default=2,
                    help="v06：连续多少任务 competence_gap 持续高才 spawn（防噪声，默认2）")
    args = ap.parse_args()

    if args.seq == "retention":
        seqs = T.RETENTION_SEQUENCES
    elif args.seq == "equi":
        seqs = {"equi": T.EQUI_SEQUENCE}
    elif args.seq == "mini":
        seqs = {"mini": T.MINI_SEQUENCE}
    else:
        seqs = {"core": T.CORE_SEQUENCE} if args.seq == "core" else {"long": T.LONG_SEQUENCE}
    cfg = dict(hidden=32, r=16, init_nodes=12, eq_nodes=24, k_fixed=5, kmin=3, kmax=8,
               memory_max=256, plastic_lr=1e-3, plastic_cap=0.1,
               eval_every=120, spawn_cos_thr=0.55, prune_usage_thr=0.02, merge_cos_thr=0.95,
               mlp_hidden=64, lr=args.lr, wd=1e-5, n_train=args.n_train,
               epochs_per_task=args.epochs_per_task, bs=args.bs,
               # 六维动态开关（v01 只保留 Router + 结构 + 静态权重）
               use_plasticity=True, use_memory=True, dynamic_tau=True, dynamic_k=True,
               # 结构稳定化（v01 生效）
               young_tasks=2, freeze_tasks=2, merge_patience=2, merge_usage_thr=0.05)

    if args.profile == "v01":
        # GPT 审计：v0.1 Stability Patch —— 一次只动一个变量，先隔离「Router+结构+静态权重」
        cfg.update(use_plasticity=False, use_memory=False, dynamic_tau=False, dynamic_k=False,
                   spawn_cos_thr=0.5, merge_cos_thr=0.9, prune_usage_thr=0.01,
                   young_tasks=2, freeze_tasks=2, merge_patience=2, merge_usage_thr=0.05)
    elif args.profile == "v02":
        # 团队 v0.2-A：保留 v0.1 的结构稳定化补丁（task-boundary/spawn-freeze/merge-patience），
        # 恢复全部承重动态（memory/plasticity/τ/动态k）——GPT v0.1 复审指出这些是「能学会」的承重机制。
        # 一次只动一组变量：相较 v0.1 仅把 4 个动态开关打开，其余（稳定化 controller 配置）与 v0.1 相同。
        cfg.update(use_plasticity=True, use_memory=True, dynamic_tau=True, dynamic_k=True,
                   spawn_cos_thr=0.5, merge_cos_thr=0.9, prune_usage_thr=0.01,
                   young_tasks=2, freeze_tasks=2, merge_patience=2, merge_usage_thr=0.05)
    elif args.profile == "v02b":
        # 团队 v0.2-B（对照）：在 v02 基础上加「仅首任务」warm-up（破 A 冷启动）。
        # 已实测：A=0.455，但 B/C 仍低（后续任务没 warm，还是硬路由冷启动）。
        cfg.update(use_plasticity=True, use_memory=True, dynamic_tau=True, dynamic_k=True,
                   spawn_cos_thr=0.5, merge_cos_thr=0.9, prune_usage_thr=0.01,
                   young_tasks=2, freeze_tasks=2, merge_patience=2, merge_usage_thr=0.05,
                   warmup_steps=1, warm_temp=3.0, warm_epochs=args.warm_epochs, warm_mode="first")
    elif args.profile == "v02c":
        # 团队 v0.2-C（方案 A，本实验主体）：per-task warm-up —— 每个任务前 warm_epochs 个 epoch
        # 全班 soft 路由（Router 学到本任务亲和、每个 Node 都拿到梯度、生成本任务的 Node 组），之后 hard top-k。
        # 目标是让 B/C 也长出独立 Node 组、解决 cross-task 不足。
        cfg.update(use_plasticity=True, use_memory=True, dynamic_tau=True, dynamic_k=True,
                   spawn_cos_thr=0.5, merge_cos_thr=0.9, prune_usage_thr=0.01,
                   young_tasks=2, freeze_tasks=2, merge_patience=2, merge_usage_thr=0.05,
                   warmup_steps=1, warm_temp=3.0, warm_epochs=args.warm_epochs, warm_mode="every")
    elif args.profile == "v03":
        # 团队 v0.3（GPT 新一轮审计建议）：Protected Expert Formation
        # A. 新 Node 专属 warm-up：forward 内 per-node warm（未成熟 Node 高门控吸梯度），成熟 Node protected（hard）
        # B. Competence Lock：update_lifecycle 按 loss/entropy/usage 升 maturity，成熟 Node 可塑衰减
        # C. Re-activation：reactivate_score 找到成熟 A-Node 直接复用（run_one 已记录）
        cfg.update(use_plasticity=True, use_memory=True, dynamic_tau=True, dynamic_k=True,
                   spawn_cos_thr=0.5, merge_cos_thr=0.9, prune_usage_thr=0.01,
                   young_tasks=2, freeze_tasks=2, merge_patience=2, merge_usage_thr=0.05,
                   warmup_steps=0, warm_temp=3.0, use_lifecycle=True)
    elif args.profile == "v04":
        # 团队 v0.4（本实验主体）：v0.3 + 任务亲和约束（强制不相交）
        # 每个任务 spawn 专属 Node 组（task_owner=当前任务），路由只允许「本任务 Node + 未归属 Node」，
        # 归属其他任务的 Node 被屏蔽（protected）→ A/B/C 各自独占 Node 组，实现真隔离。
        cfg.update(use_plasticity=True, use_memory=True, dynamic_tau=True, dynamic_k=True,
                   spawn_cos_thr=0.5, merge_cos_thr=0.9, prune_usage_thr=0.01,
                   young_tasks=2, freeze_tasks=2, merge_patience=2, merge_usage_thr=0.05,
                   warmup_steps=0, warm_temp=3.0, use_lifecycle=True,
                   task_constraint=True, init_per_task=args.init_per_task)
    elif args.profile == "v04lite":
        # 团队 v0.4-lite（修正方向）：v0.3 + 软任务亲和（soft task bias）
        # 不硬隔离（v0.4 已反证），改为给 query 加可学习任务偏移 q += w*task_emb[task]，
        # 软性鼓励不同任务偏向不同 Node 组、但允许跨任务复用（保住 A/A' 调用）。task_constraint=False。
        cfg.update(use_plasticity=True, use_memory=True, dynamic_tau=True, dynamic_k=True,
                   spawn_cos_thr=0.5, merge_cos_thr=0.9, prune_usage_thr=0.01,
                   young_tasks=2, freeze_tasks=2, merge_patience=2, merge_usage_thr=0.05,
                   warmup_steps=0, warm_temp=3.0, use_lifecycle=True,
                   task_constraint=False, n_tasks=len(T.CORE_SEQUENCE),
                   soft_task_bias=args.soft_task_bias)
    elif args.profile == "v06":
        # 团队 v0.6（GPT 二审判定）：Node autonomous specialization（Dynamic Neural Ecology）
        # 核心：Novelty-triggered Spawn（Node 级判据 use_node_spawn）+ 保留 v0.3 的 per-node warm / Competence Lock /
        #       Re-activation。不再加 task embedding oracle（GPT 明令），不硬隔离，keep 软任务亲和保留作为对照可关。
        cfg.update(use_plasticity=True, use_memory=True, dynamic_tau=True, dynamic_k=True,
                   spawn_cos_thr=0.5, merge_cos_thr=0.9, prune_usage_thr=0.01,
                   young_tasks=2, freeze_tasks=2, merge_patience=2, merge_usage_thr=0.05,
                   warmup_steps=0, warm_temp=3.0, use_lifecycle=True,
                   task_constraint=False, n_tasks=len(T.CORE_SEQUENCE),
                   soft_task_bias=0.0,          # v0.6 用 Node 级 spawn，不靠 task_emb（GPT：不加 oracle）
                   use_node_spawn=True,          # Node 级 Novelty-Spawn 判据
                   spawn_patience=args.spawn_patience)
    cfg["profile"] = args.profile

    runs = ["A", "B", "D", "C"] if args.run == "all" else [args.run]
    t0 = time.time()
    results = []
    for seq_name, seq in seqs.items():
        seq_results = []
        for run in runs:
            r = run_one(run, seq, cfg.copy(), args.seed)
            r["seq"] = seq
            r["wall_sec"] = round(time.time() - t0, 1)
            seq_results.append(r)
            print(json.dumps(r, ensure_ascii=False, indent=2)[:400], flush=True)
        results.append({"seq_name": seq_name, "sequence": seq, "results": seq_results})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": cfg, "retention": args.seq == "retention", "results": results}, f, ensure_ascii=False, indent=2)
    print("SAVED:", args.out, flush=True)


if __name__ == "__main__":
    main()
