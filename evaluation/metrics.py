"""evaluation/metrics.py — FDN-v0 评估指标：遗忘、Node 激活重合、Node→任务亲和、参数漂移。

约定：所有指标以「原始可观测数据」为输入，返回标量/结构，供 JSON 直接落盘。
"""
import numpy as np


def accuracy(pred, target, tol=0.08):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return float((np.abs(pred - target) <= tol).mean())


def forgetting(acc_initial, acc_after):
    """1 - acc_after/acc_initial（initial 可能为 0，做保护）。"""
    if acc_initial <= 0:
        return 1.0 if acc_after < 1e-6 else 0.0
    return float(max(0.0, 1.0 - acc_after / acc_initial))


def set_iou(a, b):
    """激活 Node 集合 A 与 B 的 IoU。空集保护。"""
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def node_task_affinity(node_activations, n_nodes):
    """给定 {task: set(node_id)}，返回每个任务的主要 Node 组；衡量任务间是否用不相交的 Node 组。
    返回 (subsets, disjoint_frac)：subsets={task:list}；disjoint_frac=两两无交集的比例。
    """
    tasks = list(node_activations.keys())
    subsets = {t: sorted(node_activations[t]) for t in tasks}
    pairs = 0
    disjoint = 0
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            pairs += 1
            if not set(subsets[tasks[i]]) & set(subsets[tasks[j]]):
                disjoint += 1
    return subsets, (disjoint / pairs if pairs else 1.0)


def param_drift(params_before, params_after):
    """相对参数漂移 ||Δθ|| / ||θ_before||（逐片段，跳过不同形状）。"""
    num = 0.0
    den = 0.0
    for k in set(params_before) & set(params_after):
        pb = np.asarray(params_before[k], dtype=np.float32)
        pa = np.asarray(params_after[k], dtype=np.float32)
        if pb.shape != pa.shape or pb.size == 0:
            continue
        num += float(np.sum((pa - pb) ** 2))
        den += float(np.sum(pb ** 2))
    if den <= 0:
        return 0.0
    return float(np.sqrt(num) / np.sqrt(den))
