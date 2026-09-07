"""参数 scaling / Pareto frontier：TF vs FusedFW+FFN。

GPT 审计指出：d3c68bb 对比把「参数更少 + 计算更少」混为一个变量，FusedFW 的"快"不能归因架构效率。
本实验按 GPT 规格：扫参数量（200K→600K），让两边参数量尽量匹配，同数据/同steps/同seed/同vocab/同seq/同optimizer-lr，
5 seed，记录：val loss、训练时间、推理时间、实际 FLOPs(分析式)、参数量 → 找能力 vs 成本 Pareto frontier。

方法：
  - 每个目标参数量 P，对每个架构二分搜索「宽度 D」使 np() 最接近 P（匹配参数，非"强制等参单点"，是缩放曲线）
  - TF:   D 可变, n_layer=2, FFN=4x, n_head=4
  - FusedFW+FFN: D 可变, N=4D(随宽度缩放), k=16, FFN=4x, 1层(无dec投影)
  - FLOPs 按前向 MACs 分析式计（硬件无关）：TF=2层*(12D^2+2TD)*BT + head；FusedFW=enc(DN)+rho(ND)+mem(ND)+out(DD)+ffn(8D^2)+head
  - 同数据(/data/bdh/input.txt 字符级)、同预算 250 iter、5 seed
"""
import sys; sys.path.insert(0,'/tmp')
import torch, numpy as np, time
from diag_gap import FusedFW, TF, run, load_data

TARGETS=[200_000,300_000,400_000,500_000,600_000]
SEEDS=[0,1,2,3,4]

def search_width(mk, P, lo=16, hi=512):
    """二分搜索宽度D使参数量最接近P。mk(D)->模型实例（未训练）。"""
    best=None
    for D in range(lo,hi+1,4):
        m=mk(D); p=m.np()
        if best is None or abs(p-P)<abs(best[1]-P): best=(D,p)
    return best

def mk_tf(D):
    return TF(D=D,n_layer=2)
def mk_fw(D):
    return FusedFW(D=D,N=4*D,k=16,use_ffn=True)

def flops_tf(D,T=256,nl=2,vocab=256):
    # 每token每层: qkv 3D^2 + QK^T TD + attnV TD + proj D^2 + ffn 8D^2 = 12D^2+2TD ; nl层
    return nl*T*(12*D*D+2*T*D)+T*D*vocab
def flops_fw(D,N,T=256,vocab=256):
    # enc DN + rho ND + mem ND + out DD + ffn 8D^2 + head D*vocab
    return T*(D*N+N*D+N*D+D*D+8*D*D)+T*D*vocab

def report(name, mk, flops_fn, P, D):
    m=mk(D); vals=[];tr=[];ti=[];params=m.np()
    for sd in SEEDS:
        vl,trl,tt,it,np_=run(lambda: mk(D),seed=sd); vals.append(vl);tr.append(tt);ti.append(it)
    fl=flops_fn(D)
    return dict(name=name,P=int(P),D=int(D),params=int(params),val=float(np.mean(vals)),
                val_std=float(np.std(vals)),train=float(np.mean(tr)),infer=float(np.mean(it)),
                flops=float(fl))

if __name__=='__main__':
    print("=== 参数 scaling / Pareto：TF vs FusedFW+FFN（5 seed，同数据+同预算250iter）===")
    rows=[]
    for P in TARGETS:
        d_tf,p_tf=search_width(mk_tf,P)
        d_fw,p_fw=search_width(mk_fw,P)
        r_tf=report("TF",mk_tf,flops_tf,p_tf,d_tf)
        r_fw=report("FusedFW+FFN",mk_fw,lambda D: flops_fw(D,4*D),p_fw,d_fw)
        rows+= [r_tf,r_fw]
        print(f"\n--- 目标 {P//1000}K ---")
        print(f"  {r_tf['name']:<12} D={r_tf['D']:>4} 参={r_tf['params']:,} val={r_tf['val']:.3f}±{r_tf['val_std']:.3f} "
              f"训练={r_tf['train']:.1f}s 推理={r_tf['infer']*1000:.0f}ms FLOPs/fwd={r_tf['flops']/1e6:.1f}M")
        print(f"  {r_fw['name']:<12} D={r_fw['D']:>4} 参={r_fw['params']:,} val={r_fw['val']:.3f}±{r_fw['val_std']:.3f} "
              f"训练={r_fw['train']:.1f}s 推理={r_fw['infer']*1000:.0f}ms FLOPs/fwd={r_fw['flops']/1e6:.1f}M")
    import json
    with open('/tmp/param_scaling.json','w') as f: json.dump(rows,f,indent=2)
    print("\n=== Pareto：以参数为横轴、FLOPs为代价、val为能力 ===")
    print(f"  {'参数量':>8} {'TF val':>7} {'TF FLOPs':>9} | {'FW val':>7} {'FW FLOPs':>10} | 结论")
    for i in range(0,len(rows),2):
        a=rows[i]; b=rows[i+1]
        # FW 是否同等参数量下能力更优(更小val)且更省FLOPs
        better = b['val']<a['val'] and b['flops']<a['flops']
        note = "FW胜(Pareto优)" if better else ("FW能力优" if b['val']<a['val'] else "FW能力差" if b['val']>a['val']+0.02 else "≈")
        print(f"  {a['params']:>8,} {a['val']:>7.3f} {a['flops']/1e6:>8.0f}M | {b['val']:>7.3f} {b['flops']/1e6:>8.0f}M | {note}")
