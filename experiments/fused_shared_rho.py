"""方向B探针：BDH式「共享稀疏 latent + rho 层间传递」vs 「每层独立全投影」(FusedDeep) vs TF。

关键差异（方向B = 共享 + 廉价层间传递）：
  - enc(D->N)/dec(N->D)/out(D->D) 是【一套共享】权重，每个 pass 复用同一组
    -> 参数【不随深度增长】（对比 FusedDeep：每层独立 enc/dec/out，深度线性加参/加算）。
  - 深度(n_pass) = rho 状态在层间 Hebbian 累积（BDH: rho = rho + act*v，状态存边权），
    且用 topk gather 只在 k 个激活通道上做累积与读回 -> 每 pass 廉价。
  - 这是「状态在层间累积，而非每层独立全投影」。

判据（父目标：能力不输 + 训练/推理成本双降）：
  - 能力：val loss 逼近/不输 TF（同数据+同预算）。
  - 训练/推理成本：随 n_pass 增长【次线性】（共享投影复用），且能力缺口被补上。
  - 诚实：CPU 无稀疏内核，稀疏降算可能退化为稠密 -> 如实记录。

对比维度（同 D=128，k=16，N=512，3 seed，同预算 250 iter）：
  - TF 2层           （参考基线）
  - FusedDeep nl=2/4  （每层独立全投影——显示"加深线性烧钱"的旧问题）
  - FusedSharedRho n_pass=2/4/8 （方向B：共享骨架+rho层间传递——"深度不再线性烧钱"）
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, numpy as np

def load_data(path='/data/bdh/input.txt'):
    return np.memmap(path, dtype=np.uint8, mode='r')

def get_batch(data, block=256, batch=16, rng=np.random.RandomState(0), split='train'):
    n=len(data); cut=int(0.9*n); d=data[:cut] if split=='train' else data[cut:]
    ix=rng.randint(0,len(d)-block,(batch,))
    return (torch.stack([torch.from_numpy(d[i:i+block].astype(np.int64)) for i in ix]),
            torch.stack([torch.from_numpy(d[i+1:i+1+block].astype(np.int64)) for i in ix]))

# ================= 方向B：共享稀疏 latent + rho 层间传递 =================
class FusedSharedRho(nn.Module):
    """共享骨架（enc/dec/out 一套）+ rho(快权重状态) 层间 Hebbian 累积。
    每 pass 只触碰 k 个激活通道（topk gather），廉价。参数不随 n_pass 增长。
    """
    def __init__(s, D=128, N=512, k=16, n_pass=2, vocab=256):
        super().__init__(); s.D=D; s.N=N; s.k=k; s.n_pass=n_pass; s.vocab=vocab
        s.e=nn.Embedding(vocab,D)
        s.enc=nn.Linear(D,N,bias=False)      # 共享（每次 pass 复用）
        s.dec=nn.Linear(N,D,bias=False)      # 共享
        s.out=nn.Linear(D,D,bias=False)      # 共享
        s.ln=nn.LayerNorm(D)
        s.head=nn.Linear(D,vocab)
        s.alpha=nn.Parameter(torch.tensor(0.5))   # rho 累积增益
        s.gate=nn.Parameter(torch.zeros(k))       # 每激活通道门控（可学）
    def forward(s,x,t=None):
        B,T=x.size(); D=s.D; N=s.N; k=s.k
        h=s.e(x)                       # B,T,D
        ln_h=s.ln(h)
        lat=s.enc(ln_h)                # B,T,N （共享 enc，只投影一次）
        topv,topi=torch.topk(lat,k,dim=-1)
        act_val=torch.relu(topv)*torch.sigmoid(s.gate).view(1,1,k)  # B,T,k
        sp=float(k)/float(N)  # 真实稀疏度：激活通道占全部 latent 的比例（k/N）
        # 稀疏索引扁平化（scatter 用）：把 (t,j) 摊到 S=T*k，沿 dim=1 (N 通道) 累积
        topi_flat=topi.view(B,T*k)                                  # B,S
        idx1=topi_flat.unsqueeze(-1).expand(B,T*k,D).contiguous()   # B,S,D
        act_flat=act_val.reshape(B,T*k)                             # B,S
        ln_rep=ln_h.unsqueeze(2).expand(B,T,k,D).reshape(B,T*k,D)   # B,S,D
        # rho 状态（B,N,D），在层间 Hebbian 累积（BDH: rho += alpha*act*v）
        rho=torch.zeros(B,N,D,device=x.device)
        for _ in range(s.n_pass):
            contrib=torch.zeros(B,N,D,device=x.device).scatter_add(
                1, idx1, s.alpha*act_flat.unsqueeze(-1)*ln_rep)
            rho=rho+contrib                                          # 跨 pass 累积（非原地）
            # 从记忆读回上下文（只 k 通道）：ctx=sum_j act_val*rho[topi]
            ctx=torch.einsum('btk,btkd->btd', act_val,
                             rho[torch.arange(B,device=x.device).view(B,1,1), topi])
            # 稀疏投影读回（共享 dec，只取 k 激活行）-> B,T,D
            decT=s.dec.weight.T[topi]          # B,T,k,D
            proj=torch.einsum('btk,btkd->btd', act_val, decT)
            h=h+proj+ctx
        h=h+s.out(s.ln(h))
        lg=s.head(s.ln(h))
        loss=None
        if t is not None: loss=F.cross_entropy(lg.view(-1,s.vocab),t.view(-1))
        return lg,loss,sp,rho
    def np(self): return sum(p.numel() for p in self.parameters())

# ================= 每层独立全投影（旧问题：加深线性烧钱） =================
class FusedLayer(nn.Module):
    def __init__(s,D=128,N=512,k=16):
        super().__init__(); s.D=D; s.N=N; s.k=k
        s.enc=nn.Linear(D,N,bias=False); s.dec=nn.Linear(N,D,bias=False)
        s.out=nn.Linear(D,D,bias=False); s.ln1=nn.LayerNorm(D); s.ln2=nn.LayerNorm(D)
    def forward(s,x):
        B,T,D=x.size(); N=s.N; k=s.k
        et=s.ln1(x); lat=s.enc(et)
        topv,topi=torch.topk(lat,k,dim=-1)
        act=torch.relu(torch.zeros_like(lat).scatter(-1,topi,topv))
        h=et+s.dec(act); h=s.ln2(h); return x+h
    def np(self): return sum(p.numel() for p in self.parameters())
class FusedDeep(nn.Module):
    def __init__(s,D=128,N=512,k=16,n_layer=2,vocab=256):
        super().__init__(); s.D=D; s.N=N; s.k=k; s.n_layer=n_layer; s.vocab=vocab
        s.e=nn.Embedding(vocab,D)
        s.layers=nn.ModuleList([FusedLayer(D,N,k) for _ in range(n_layer)])
        s.ln=nn.LayerNorm(D); s.head=nn.Linear(D,vocab)
    def forward(s,x,t=None):
        h=s.e(x)
        for l in s.layers: h=l(h)
        lg=s.head(s.ln(h)); loss=None
        if t is not None: loss=F.cross_entropy(lg.view(-1,s.vocab),t.view(-1))
        return lg,loss
    def np(self): return sum(p.numel() for p in self.parameters())

# ================= TF（参考基线，2层D=128） =================
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
    t0=time.time()
    sp=0.0
    for _ in range(iters):
        m.train(); x,y=get_batch(data,bl,bt,rng)
        if isinstance(m,FusedSharedRho): _,loss,sp,_=m(x,y)
        else: _,loss=m(x,y)
        opt.zero_grad(); loss.backward(); opt.step()
    train_t=time.time()-t0
    m.eval()
    with torch.no_grad():
        xv,yv=get_batch(data,bl,8,rng,split='val')
        if isinstance(m,FusedSharedRho): _,vl,sp,_=m(xv,yv)
        else: _,vl=m(xv,yv)
        x,_=get_batch(data,bl,8,rng); t1=time.time()
        for _ in range(3):
            if isinstance(m,FusedSharedRho): m(x)
            else: m(x)
        infer_t=(time.time()-t1)/3
    return vl.item(),train_t,infer_t,m.np(),sp

if __name__=='__main__':
    seeds=[0,1,2]
    def report(name,fn):
        vals=[];trs=[];ins=[];sps=[]
        for sd in seeds:
            vl,tr,inf,np_,sp=fn(sd); vals.append(vl);trs.append(tr);ins.append(inf);sps.append(sp)
        print(f" {name:<22} val={np.mean(vals):.3f} 训练={np.mean(trs):.1f}s 推理={np.mean(ins)*1000:.0f}ms 参数={np_:,} 稀疏={np.mean(sps):.1%}")
        return np.mean(vals),np.mean(trs)
    print("=== 方向B：共享稀疏latent + rho层间传递 vs 每层独立投影 vs TF（同数据+同预算，3seed）===")
    report("TF(2层D=128)",lambda sd: run(lambda: TF(D=128,n_layer=2),seed=sd))
    report("FusedDeep nl=2",lambda sd: run(lambda: FusedDeep(D=128,N=512,k=16,n_layer=2),seed=sd))
    report("FusedDeep nl=4",lambda sd: run(lambda: FusedDeep(D=128,N=512,k=16,n_layer=4),seed=sd))
    for np_ in [2,4,8]:
        report(f"SharedRho pass={np_}",lambda sd,np_=np_: run(lambda: FusedSharedRho(D=128,N=512,k=16,n_pass=np_),seed=sd))
