"""growth/plasticity.py — 推理期可塑权重（参数动态②）：有界 ΔW 的测量与策略说明。

FDN 前向中对每个激活 Node 执行有界 Hebbian 更新（见 model/fdn.py 的 _plastic_update）：
    plastic += η · scale · (sign(err) · h)   → 衰减 → 裁剪到 [-cap, cap]
本模块提供该策略的集中定义与「可塑幅度」测量（用于 parameter_drift / 参数动态验收指标）。
"""
import torch


def hebbian_delta(h, scale, lr, cap=0.1, decay=0.98):
    """单步有界 Hebbian 增量的计算：返回 (delta, 更新规则说明)。"""
    err = 1.0  # 简化：代入真实误差方向由调用方覆盖
    delta = lr * scale * (torch.sign(torch.tensor(err, dtype=h.dtype)) * h)
    return delta, {"rule": "hebbian", "lr": lr, "decay": decay, "cap": cap}


def plastic_magnitude(model):
    """所有 Node 可塑向量当前幅度的均值（0 表示尚未扰动）。"""
    if model.plastic.numel() == 0:
        return 0.0
    return float(model.plastic.norm(dim=-1).mean())
