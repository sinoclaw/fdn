"""确认：在 rho 读回器上补 FFN 能否把能力差距闭合，且保住效率优势。
5 seed 稳定性确认（吸取单 seed 峰值不可信教训）。

配置（同数据+同预算250iter+5 seed）：
  TF 2层            = 参考基线
  FusedFW(现状)      = 基线（差距基准）
  FusedFW + FFN      = 在 rho 读回器后补 FFN（非线性重混合容量）

判据（跑前锁死）：
  - 能力：FusedFW+FFN 的 val 差距相对 TF 显著小于 FusedFW，且 5 seed 稳定（std小、方向一致）
  - 效率：训练/推理仍显著快于 TF，参数少于 TF
  - 目标：验证"补 FFN 是补能力差距的最便宜杠杆"是否稳定成立
"""
import sys; sys.path.insert(0,'/tmp')
import numpy as np, torch, time
from diag_gap import FusedFW, TF, run, load_data, get_batch

SEEDS=[0,1,2,3,4]
def report(name, mk):
    vs=[];tr=[];ti=[];trls=[]
    for sd in SEEDS:
        vl,trl,tt,it,np_=run(mk,seed=sd); vs.append(vl);tr.append(tt);ti.append(it);trls.append(trl)
    print(f" {name:<22} val={np.mean(vs):.3f}±{np.std(vs):.3f} 训末loss={np.mean(trls):.3f} "
          f"训练={np.mean(tr):.1f}s 推理={np.mean(ti)*1000:.0f}ms 参数={np_:,}")
    return np.mean(vs), np.std(vs)

if __name__=='__main__':
    print("=== 确认：rho 读回器补 FFN 能否闭合能力差距（5 seed）===")
    tf_m,tf_s=report("TF 2层(D=128)",lambda: TF(D=128,n_layer=2))
    fw_m,fw_s=report("FusedFW 1层(现状)",lambda: FusedFW(D=128,N=512,k=16))
    ff_m,ff_s=report("FusedFW 1层+FFN",lambda: FusedFW(D=128,N=512,k=16,use_ffn=True))
    print("\n=== 能力差距（相对 TF，5 seed 均值）===")
    print(f"  FusedFW 现状 : {fw_m-tf_m:+.3f}  (std {fw_s:.3f})")
    print(f"  FusedFW +FFN : {ff_m-tf_m:+.3f}  (std {ff_s:.3f})")
    print("\n=== 判据 ===")
    print(f" ①能力更接近: {abs(ff_m-tf_m) < abs(fw_m-tf_m)} ({abs(fw_m-tf_m):.3f} -> {abs(ff_m-tf_m):.3f})")
    print(" ②5 seed 稳定: FusedFW+FFN std=%.3f"%ff_s)
