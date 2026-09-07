"""长 seq 能力 5 seed 定论：TF vs FusedFW+FFN（参数匹配 D=128 vs D=156）。
把长 seq 能力抽查（80 iter 单 seed 趋势）升为 5 seed 定论：长上下文下 FW 能力是否仍持平/略优。

同预算（TF/FW 同 iters/batch/seq/lr/seed）、同数据、5 seed，报 val mean±std。
seq = block 长度（即 LM 上下文窗口）。
"""
import sys; sys.path.insert(0,'/tmp')
import torch, numpy as np, time
from diag_gap import load_data, get_batch
from long_seq import TFLong
from diag_gap import FusedFW

SEEDS=[0,1,2,3,4]
ITERS=120
BATCH=8

def run_at(mk, T, seed, iters=ITERS, bt=BATCH):
    # P0 fix(GPT审计 2026-09-07): 先设 seed 再建模型
    data=load_data(); rng=np.random.RandomState(seed); torch.manual_seed(seed); np.random.seed(seed)
    m=mk()
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
    for _ in range(iters):
        m.train(); x,y=get_batch(data,T,bt,rng); _,loss=m(x,y); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): xv,yv=get_batch(data,T,4,rng,split='val'); _,vl=m(xv,yv)
    return vl.item()

def report(name, mk, T):
    vals=[run_at(mk,T,sd) for sd in SEEDS]
    print(f"  T={T:>5} {name:<8} val={np.mean(vals):.3f}±{np.std(vals):.3f}  (seeds {[round(v,3) for v in vals]})")
    return np.mean(vals), np.std(vals)

if __name__=='__main__':
    print(f"=== 长 seq 能力 5 seed 定论（同预算 iters={ITERS}, batch={BATCH}, 5 seed）===")
    for T in [512,1024]:
        print(f"\n--- seq={T} ---")
        tf=report("TF",lambda: TFLong(D=128,n_layer=2),T)
        fw=report("FW+FFN",lambda: FusedFW(D=156,N=4*156,k=16,use_ffn=True),T)
        d=fw[0]-tf[0]
        print(f"  → FW 差距={d:+.3f}  ({'FW优/持平' if d<=0.02 else 'FW差'})")
