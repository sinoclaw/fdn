"""能力差距诊断：FusedFW(单层无注意力无FFN) vs 加容量消融 vs TF。
定位"能力差0.26"的根因——是容量/表达能力不足，还是 rho 机制本身弱。

一次只动一组变量：
  TF 2层                  = 参考基线
  FusedFW 1层(现状)        = 差距基准（单层/无注意/无FFN/rhO袋状记忆）
  FusedFW 1层+FFN          = 只加 FFN 非线性（同深度）
  FusedFW 2层(逐层独立)     = 只加深度（每层独立 enc/sparse/dec）
  FusedFW 1层 N=1024       = 只加 latent 宽度

同数据(字符级shakespeare)+同预算(250 iter)+3 seed。报告 val/训练val/参数/训练+推理时间。
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, numpy as np

def load_data(path='/data/bdh/input.txt'):
    return np.memmap(path, dtype=np.uint8, mode='r')
def get_batch(data, block=256, batch=16, rng=np.random.RandomState(0), split='train'):
    n=len(data); cut=int(0.9*n); d=data[:cut] if split=='train' else data[cut:]
    ix=rng.randint(0,len(d)-block,(batch,))
    return (torch.stack([torch.from_numpy(d[i:i+block].astype(np.int64)) for i in ix]),
            torch.stack([torch.from_numpy(d[i+1:i+1+block].astype(np.int64)) for i in ix]))

# 基础稀疏+rho 单元
def sparse_rho_forward(self, et, enc, dec, out):
    B,T,D=et.size(); N=self.N; k=self.k
    lat=enc(et)                      # B,T,N
    topv,topi=torch.topk(lat,k,dim=-1)
    act=torch.relu(torch.zeros_like(lat).scatter(-1,topi,topv))   # 稀疏正激活
    ln_et=self.ln(et)                # 归一化嵌入
    rho=torch.einsum('btn,btd->bnd', act, ln_et)     # Hebbian 累积（袋状，无时序）
    mem_ctx=torch.einsum('btn,bnd->btd', act, rho)   # act 加权读回
    h=et+mem_ctx
    h=self.ln(h)
    h=out(h)                          # D->D
    return h, act, rho

class FusedFW(nn.Module):
    """现状：单层，无注意力，无FFN。"""
    def __init__(s,D=128,N=512,k=16,vocab=256,use_ffn=False,n_layer=1,head=128):
        super().__init__(); s.D=D; s.N=N; s.k=k; s.vocab=vocab
        s.use_ffn=use_ffn; s.n_layer=n_layer; s.head_dim=head
        s.e=nn.Embedding(vocab,D); s.ln=nn.LayerNorm(D); s.out=nn.Linear(D,D,bias=False); s.head=nn.Linear(D,vocab)
        s.encs=nn.ModuleList([nn.Linear(D,N,bias=False) for _ in range(n_layer)])
        s.decs=nn.ModuleList([nn.Linear(N,D,bias=False) for _ in range(n_layer)])
        if use_ffn:
            s.ffn=nn.Sequential(nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D))
    def forward(s,x,t=None):
        B,T=x.size(); h=s.e(x)
        for i in range(s.n_layer):
            h,_,_=sparse_rho_forward(s,h,s.encs[i],s.decs[i],s.out)
            if s.use_ffn: h=h+s.ffn(s.ln(h))
        lg=s.head(s.ln(h)); loss=None
        if t is not None: loss=F.cross_entropy(lg.view(-1,s.vocab),t.view(-1))
        return lg,loss
    def np(self): return sum(p.numel() for p in self.parameters())

class TF(nn.Module):
    def __init__(s,D=128,nh=4,n_layer=2,vocab=256):
        super().__init__(); s.vocab=vocab; s.D=D; s.nh=nh
        s.e=nn.Embedding(vocab,D); s.p=nn.Parameter(torch.zeros((1,512,D)))
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

def run(factory,iters=250,bl=256,bt=16,lr=3e-4,seed=0):
    # P0 fix(GPT审计 2026-09-07): 必须先设 seed 再创建模型，否则模型随机初始化不受 seed 控制
    rng=np.random.RandomState(seed); torch.manual_seed(seed); np.random.seed(seed)
    data=load_data()
    m=factory()
    opt=torch.optim.AdamW(m.parameters(),lr=lr)
    t0=time.time(); trl=[]
    for _ in range(iters):
        m.train(); x,y=get_batch(data,bl,bt,rng); _,loss=m(x,y)
        opt.zero_grad(); loss.backward(); opt.step(); trl.append(loss.item())
    train_t=time.time()-t0
    m.eval()
    with torch.no_grad():
        xv,yv=get_batch(data,bl,8,rng,split='val'); _,vl=m(xv,yv)
        x,_=get_batch(data,bl,8,rng); t1=time.time()
        for _ in range(3): m(x)
        infer_t=(time.time()-t1)/3
    return vl.item(),np.mean(trl[-20:]),train_t,infer_t,m.np()

if __name__=='__main__':
    seeds=[0,1,2]
    def report(name,fn):
        vs=[];tr=[];ti=[];trls=[]
        for sd in seeds:
            vl,trl,tt,it,np_=fn(sd); vs.append(vl);tr.append(tt);ti.append(it);trls.append(trl)
        print(f" {name:<24} val={np.mean(vs):.3f} 训末loss={np.mean(trls):.3f} 训练={np.mean(tr):.1f}s 推理={np.mean(ti)*1000:.0f}ms 参数={np_:,}")
        return np.mean(vs)
    print("=== 能力差距诊断：FusedFW vs 加容量消融 vs TF（同数据+同预算250iter+3seed）===")
    r_tf=report("TF 2层(D=128)",lambda sd: run(lambda: TF(D=128,n_layer=2),seed=sd))
    r_fw=report("FusedFW 1层(现状)",lambda sd: run(lambda: FusedFW(D=128,N=512,k=16),seed=sd))
    r_ffn=report("FusedFW 1层+FFN",lambda sd: run(lambda: FusedFW(D=128,N=512,k=16,use_ffn=True),seed=sd))
    r_2l=report("FusedFW 2层(独立)",lambda sd: run(lambda: FusedFW(D=128,N=512,k=16,n_layer=2),seed=sd))
    r_wide=report("FusedFW 1层 N=1024",lambda sd: run(lambda: FusedFW(D=128,N=1024,k=16),seed=sd))
    print("\n=== 能力差距（相对 TF）===")
    for nm,v in [("FusedFW 1层",r_fw),("+FFN",r_ffn),("2层",r_2l),("N=1024",r_wide)]:
        print(f"  {nm:<16} 差={v-r_tf:+.3f}")
