"""tasks.py — FDN-v0 的 toy 矢量任务生成器。

各任务用「输入分布 + 目标函数」区分，Router 从输入自身学习任务亲和（不喂 oracle task-id）。
统一输入维度 DIM；目标为归一化标量 y∈[0,1]；回归用 MSE，准确率用 |pred-y|<=tol。
"""
import numpy as np

DIM = 4  # 统一输入维度（add/mul 用前2维，logic 前4维，seq 前3维）


def _norm_x(vec, dim=DIM):
    v = np.asarray(vec, dtype=np.float32).ravel()
    if v.size < dim:
        v = np.concatenate([v, np.zeros(dim - v.size, dtype=np.float32)])
    return v.astype(np.float32)


def gen_add(n, seed=0, lo=0.0, hi=1.0, shift=0.0):
    """A_add: x=(a,b)∈[lo,hi]^2, y=(a+b)/2 (∈[0,1])。shift 用于 A' 变体（规则未变，分布平移）。"""
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for _ in range(n):
        a = rng.uniform(lo, hi) + shift
        b = rng.uniform(lo, hi) + shift
        xs.append(_norm_x([a, b]))
        ys.append(float((a + b) / 2.0))
    return np.stack(xs), np.array(ys, dtype=np.float32)


def gen_mul(n, seed=0, lo=0.5, hi=1.0):
    """B_mul: x=(a,b)∈[lo,hi]^2, y=a*b。用与 add 不同的 support 让输入分布可区分。"""
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for _ in range(n):
        a = rng.uniform(lo, hi)
        b = rng.uniform(lo, hi)
        xs.append(_norm_x([a, b]))
        ys.append(float(a * b))
    return np.stack(xs), np.array(ys, dtype=np.float32)


def gen_logic(n, seed=0):
    """C_logic: x∈{0,1}^4, y=majority/4。位可以是 0/1（转 float），目标为多数占比。"""
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for _ in range(n):
        bits = rng.randint(0, 2, size=4).astype(np.float32)
        xs.append(_norm_x(bits))
        ys.append(float(bits.mean()))  # 多数 → 归一化标量
    return np.stack(xs), np.array(ys, dtype=np.float32)


def gen_seq(n, seed=0, dmin=0.05, dmax=0.25):
    """D_seq: x=[v0,v1,v2] 等差三元组，y=下一项 v3（归一化）。"""
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for _ in range(n):
        v0 = rng.uniform(0.0, 0.6)
        d = rng.uniform(dmin, dmax)
        v1, v2 = v0 + d, v0 + 2 * d
        v3 = v0 + 3 * d
        vn = np.clip([v0, v1, v2], 0.0, 1.0)
        xs.append(_norm_x(vn))
        ys.append(float(np.clip(v3, 0.0, 1.0)))
    return np.stack(xs), np.array(ys, dtype=np.float32)


# 任务注册表：给出核心实验序列 A→B→C→A'
TASKS = {
    "A_add":  lambda n, s: gen_add(n, s),
    "B_mul":  lambda n, s: gen_mul(n, s),
    "C_logic": lambda n, s: gen_logic(n, s),
    "D_seq":  lambda n, s: gen_seq(n, s),
    "A2_add": lambda n, s: gen_add(n, s, lo=0.15, hi=0.85, shift=0.0),  # A' 变体
}

# 核心实验顺序
CORE_SEQUENCE = ["A_add", "B_mul", "C_logic", "A2_add"]
LONG_SEQUENCE = ["A_add", "B_mul", "C_logic", "A2_add", "D_seq", "B_mul"]

# 各任务用来算精度的容差
TOL = 0.08
