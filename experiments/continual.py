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


def evaluate_model(model, seq_name, seed):
    """对每个任务生成评估集并测准确率 + 记录激活 Node。"""
    accs = {}
    activations = {}
    for tname in seq_name:
        x, y = T.TASKS[tname](512, seed)
        pred, info = predict(model, torch.from_numpy(x))
        accs[tname] = M.accuracy(pred, y, T.TOL)
        if "idx" in info:
            activations[tname] = sorted(set(info["idx"].reshape(-1).tolist()))
    return accs, activations


def train_task(model, x, y, optimizer, epochs, bs, dynamic, warm_epochs=0):
    model.train()
    lossf = nn.MSELoss()
    for ep in range(epochs):
        # 方案 A：per-task warm-up —— 每个任务的前 warm_epochs 个 epoch 用全班 soft 路由，
        # 让本任务的新分布能路由到 Node、每个 Node 都拿到梯度（生成本任务的 Node 组），之后切 hard top-k。
        in_warm = warm_epochs > 0 and ep < warm_epochs
        if in_warm:
            model.warmup_steps = 10 ** 9   # 超大值 → 本任务此 epoch 全程 warm
        else:
            model.warmup_steps = 0
        for xb, yb in make_batches(x, y, bs, shuffle=True):
            optimizer.zero_grad()
            res = model.forward(xb, reset=True)       # toy 独立回归：每次前向清零状态
            out = res[0] if isinstance(res, tuple) else res
            loss = lossf(out.reshape(-1), yb)
            loss.backward()
            optimizer.step()
            # FDN：训练时把 (输入, 目标) 写入外部记忆（use_memory 内部关闭则 noop）
            if dynamic and hasattr(model, "write_memory"):
                model.write_memory(xb, yb)
            # 结构演化改为「任务边界触发」，见 run_one 的 controller.task_boundary


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
                    warmup_steps=cfg.get("warmup_steps", 0), warm_temp=cfg.get("warm_temp", 3.0))
        dynamic, controller = True, EvolutionController(
            spawn_cos_thr=cfg["spawn_cos_thr"], prune_usage_thr=cfg["prune_usage_thr"],
            merge_cos_thr=cfg["merge_cos_thr"], young_tasks=cfg["young_tasks"],
            freeze_tasks=cfg["freeze_tasks"], merge_patience=cfg["merge_patience"],
            merge_usage_thr=cfg["merge_usage_thr"])

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    rec["param_count"] = model.param_count()

    timeline = []          # 每任务阶段后的全任务准确率
    all_activations = {}   # task -> set(node)
    per_task_best = {}

    for pos, task in enumerate(seq):
        x, y = T.TASKS[task](cfg["n_train"], seed + pos * 977)
        # warm 策略：warm_mode='first'=仅首任务 warm（v02b，只破 A 冷启动），'every'=每任务 warm（v02c，方案A，每个能力组都长出）
        warm_epochs = cfg.get("warm_epochs", 0)
        if cfg.get("warm_mode") == "first" and pos > 0:
            warm_epochs = 0
        train_task(model, x, y, opt, cfg["epochs_per_task"], cfg["bs"], dynamic,
                   warm_epochs=warm_epochs)
        if dynamic and controller is not None:
            controller.task_boundary(model, opt, x)   # 结构演化：任务边界触发（降频）
        accs, act = evaluate_model(model, seq, seed + 1000 + pos)
        timeline.append({"after": task, "acc": accs})
        all_activations[task] = set(act.get(task, []))

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

    if dynamic:
        rec["final_nodes"] = model.node_count()
        rec["spawn_count"] = controller.spawn_count
        rec["prune_count"] = controller.prune_count
        rec["merge_count"] = controller.merge_count
        rec["memory_size"] = model.memory.size
        rec["plastic_magnitude"] = plastic_magnitude(model)
        rec["k_used"] = cfg["k_fixed"]
    else:
        rec["final_nodes"] = len(model.nodes) if hasattr(model, "nodes") else 1

    rec["timeline"] = timeline
    rec["final_acc"] = timeline[-1]["acc"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="core", choices=["core", "long", "retention"],
                    help="retention：跑 A→A' / A→B→A' / A→B→C→A' 三组，测 A 保留曲线")
    ap.add_argument("--out", default="results/summary_v0.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=1200)
    ap.add_argument("--epochs_per_task", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--run", default="all", choices=["all", "A", "B", "D", "C"])
    ap.add_argument("--profile", default="v0", choices=["v0", "v01", "v02", "v02b", "v02c"],
                    help="v0=全动态；v01=StabilityPatch(关承重动态+结构稳定化)；v02=稳定结构+全动态；v02b=v02+首任务warm(curriculum)；v02c=v02+每任务warm(per-task warm-up)")
    ap.add_argument("--warm_epochs", type=int, default=0,
                    help="v02c：per-task warm-up —— 每个任务前 N 个 epoch 用全班 soft 路由，之后切 hard top-k")
    args = ap.parse_args()

    if args.seq == "retention":
        seqs = T.RETENTION_SEQUENCES
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
