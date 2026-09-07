"""长 seq 缩放验证：TF（注意力 O(T^2)）vs FusedFW+FFN（rho 记忆 O(T) 线性）。

核心：随序列长度 T 增大，TF 注意力成本 ~O(T^2)（每翻倍≈4x），FusedFW 记忆 ~O(T)（每翻倍≈2x）。
扫 T ∈ {128,256,512,1024,2048,4096}（batch=1，5 次前向取均值），对比两架构前向 wall-clock 及翻倍比。

参数匹配（保持 Pareto 公平）：TF D=128（≈528K），FusedFW+FFN D=156（≈500K，同量级）。
另加：长 seq 下能力抽查（T=512/1024，短训~80 iter，看 FW 在长上下文是否仍不差）。
"""
import sys; sys.path.insert(0,'/tmp')
import torch, torch.nn as nn, torch.nn.functional as F, time, numpy as np
from diag_gap import FusedFW, TF, run, load_data, get_batch

class TFLong(nn.Module):
    """TF 支持长 seq（位置编码扩到 8192）。"""
    def __init__(s,D=128,nh=4,n_layer=2,vocab=256,max_seq=8192):
        super().__init__(); s.vocab=vocab; s.D=D; s.nh=nh
        s.e=nn.Embedding(vocab,D); s.p=nn.Parameter(torch.zeros((1,max_seq,D)))
        s.blocks=nn.ModuleList()
        for _ in range(n_layer):
            s.blocks.append(nn.ModuleList([nn.LayerNorm(D),nn.Linear(D,3*D),nn.Linear(D,D),
                nn.LayerNorm(D),nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D)]))
        s.ln=nn.LayerNorm(D); s.h=nn.Linear(D,vocab)
    def forward(s,x,t=None):
        B,T=x.size(); y=s.e(x)+s.p[:,:T]; h=s.nh; D=s.D
        for ln1,qkv,proj,ln2,w1,act,w2 in s.blocks:
            xx=ln1(y); q,k,v=qkv(xx).chunk(3,dim=-1)
            q=q.view(B,T,h,D//h).transpose(1,2);k=k.view(B,T,h,D//h).transpose(1,2);v=v.view(B,T,h,D//h).transpose(1,2)
            sc=(q@k.transpose(-2,-1))*(D//h)**-0.5
            mask=torch.tril(torch.ones(T,T)).view(1,1,T,T)
            o=torch.softmax(sc.masked_fill(mask==0,float('-inf')),-1)@v
            o=o.transpose(1,2).contiguous().view(B,T,D); o=proj(o)
            y=y+o; y=y+w2(act(w1(ln2(y))))
        lg=s.h(s.ln(y)); loss=None
        if t is not None: loss=F.cross_entropy(lg.view(-1,s.vocab),t.view(-1))
        return lg,loss
    def np(self): return sum(p.numel() for p in self.parameters())

def fwd_time(m, T, batch=1, n=5):
    x=torch.randint(0,256,(batch,T))
    m.eval()
    with torch.no_grad():
        for _ in range(2): m(x)   # warmup
        t=time.time()
        for _ in range(n): m(x)
    return (time.time()-t)/n

def scale_row(name, mk, Ts):
    print(f"\n=== {name} 前向时间随 T 缩放（batch=1）===")
    prev=None; prevT=None
    times={}
    for T in Ts:
        t=fwd_time(mk(),T)*1000  # ms
        times[T]=t
        ratio = (t/prev) if prev else float('nan')
        logratio = np.log2(ratio) if prev else float('nan')
        print(f"  T={T:>5}  {t:7.2f} ms  翻倍比={ratio:.2f}  (每倍序列 {logratio:.2f} 倍耗时; O(T^2)≈log2=2, O(T)=1)")
        prev=t; prevT=T
    return times

if __name__=='__main__':
    Ts=[128,256,512,1024,2048,4096]
    print("=== 长 seq 缩放：TF(注意力O(T^2)) vs FusedFW+FFN(rho记忆O(T)) ===")
    print("注：翻倍比=log2(时间增长)，O(T^2)注意力≈2.0，O(T)线性≈1.0")
    tf_times=scale_row("TF D=128", lambda: TFLong(D=128,n_layer=2), Ts)
    fw_times=scale_row("FusedFW+FFN D=156", lambda: FusedFW(D=156,N=4*156,k=16,use_ffn=True), Ts)
    print("\n=== 相对速度 FW/TF（<1 = FW 更快）===")
    for T in Ts:
        w=fw_times[T]; t=tf_times[T]
        print(f"  T={T:>5}  FW={w:7.2f}ms  TF={t:7.2f}ms  FW/TF={w/t:.2f}")
    print("\n▶ 若 FW/TF 随 T 递减 → 长上下文下 FusedFW 相对优势放大（O(T) 胜过 O(T^2)）")

    # 能力抽查：T=512, 1024（短训，1 seed 先看趋势）
    print("\n=== 长 seq 能力抽查（80 iter, seed 0）===")
    for T in [512,1024]:
        for name,mk in [("TF",lambda: TFLong(D=128,n_layer=2)),("FW+FFN",lambda: FusedFW(D=156,N=4*156,k=16,use_ffn=True))]:
            try:
                # 用 run 但改 block/seq：run 里 bl 默认256，这里临时用定制训练
                data=load_data(); rng=np.random.RandomState(0); torch.manual_seed(0); np.random.seed(0)
                m=mk()
                opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
                for _ in range(80):
                    m.train(); x,y=get_batch(data,T,8,rng); _,loss=m(x,y); opt.zero_grad(); loss.backward(); opt.step()
                m.eval()
                with torch.no_grad(): xv,yv=get_batch(data,T,4,rng,split='val'); _,vl=m(xv,yv)
                print(f"  T={T} {name:<7} val={vl.item():.3f}")
            except Exception as e:
                print(f"  T={T} {name:<7} 失败: {type(e).__name__}: {str(e)[:80]}")
