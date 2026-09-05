"""model/dynamic_node.py — DynamicNode：带输入调制时间常数、独立计算路径与独立输出头的动态单元。

每个 Node 是完整独立的小子网（in_proj + 循环单元 + out_head），只有被 Router 选中时才参与计算，
因此对新任务的旧任务能力是「模块隔离」的（GSLM 缺口的关键修复）。

更新式（连续时间）： h_{t+1} = h_t + Δt · tanh(W_x x + W_h h + b)，Δt = τ_i = sigmoid(U_tau [x; h])。
"""
import torch
import torch.nn as nn


class DynamicNode(nn.Module):
    def __init__(self, dim=4, hidden=32, dynamic_tau=True):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.dynamic_tau = dynamic_tau
        self.in_proj = nn.Linear(dim, hidden)          # 输入投影
        self.cell_h = nn.Linear(hidden, hidden, bias=False)  # 循环权重 W_h
        self.cell_x = nn.Linear(dim, hidden, bias=False)     # 输入权重 W_x
        self.cell_b = nn.Parameter(torch.zeros(hidden))      # 偏置
        self.tau = nn.Linear(hidden + dim, 1)               # τ = f(x, h) ∈ (0,1)
        self.tau_base = 0.5                                  # 关闭时间动态时的固定 τ
        self.out_head = nn.Linear(hidden, 1)                # 独立输出头（模块隔离关键）
        self.reset()

    def reset(self):
        with torch.no_grad():
            for m in (self.in_proj, self.cell_h, self.cell_x, self.tau, self.out_head):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        m.bias.zero_()
            self.cell_b.zero_()

    def forward(self, h, x, dt=1.0):
        """h:(B,H) 旧状态, x:(B,DIM)。返回 (h_new, out)。"""
        if self.dynamic_tau:
            ctx = torch.cat([x, h], dim=-1)
            tau = torch.sigmoid(self.tau(ctx))                 # (B,1) 输入调制时间常数
        else:
            tau = torch.full((h.shape[0], 1), self.tau_base, device=h.device)  # 固定 τ（时间动态关闭）
        f = torch.tanh(self.cell_x(x) + self.cell_h(h) + self.cell_b)  # (B,H)
        h_new = h + dt * tau * f                           # 连续时间更新
        out = self.out_head(h_new)                         # (B,1) 独立输出
        return h_new, out

    def params_dict(self):
        return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
