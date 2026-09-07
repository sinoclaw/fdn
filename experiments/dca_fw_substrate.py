"""步骤3-续：FusedFW 作为 DCA 的 Capability substrate（合体动态能力 + O(T) 高效）。

把 DCA 池的小型 MLP Cap 换成 **FusedFW 记忆单元（CapFW）**：
  - 输入：skill 特征向量 (n, FEAT_DIM=64) → reshape 成 64 个"token"、每 token 1 维
  - 前向：inp(1->D) → enc(D->N) → topk稀疏 → rho Hebbian累积 → 记忆读回 → FFN → 池化 → head(N_CLASS)
  - 这复用 FusedFW 的核心（稀疏激活 + rho 快权重记忆 + FFN），体现"无注意力 O(T) 记忆"作为能力载体。

管理：池用免训 Interference Predictor（[task_rep; cap_state(CapFW)] → damage → REUSE/SPAWN，运行时零训练）。
验证：predictor 泛化 + 池正确 REUSE/SPAWN（同族复用/异族新建）+ 旧能力不遗忘（5 seed? 先 2 seed 看方向）。

注：cap_state 与 predictor 都基于 CapFW（不是 MLP）——不同于 pool2，这次是"FusedFW 基底"的完整重导。
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, json, copy
sys.path.insert(0,'/tmp')
from dca_v1_skills import gen_classify, gen_extract, gen_math, gen_transform, FEAT_DIM
from dca_v1_ip2 import (to_class_label, N_CLASS, SKILLS, load_skill, task_rep_skill,
                        build_dataset, rankdata, spearmanr, f1_bin)
from dca_v1_ip3 import (make_many_variants, load_skillv, task_repv, probe_damagev,
                        VARIANTS, fam_of_variant, build_balanced)

# ---------- CapFW：FusedFW 式序列分类器（能力载体） ----------
class CapFW(nn.Module):
    def __init__(self, n_class=N_CLASS, dim_in=1, D=64, N=256, k=16):
        super().__init__(); self.dim_in=dim_in; self.D=D; self.N=N; self.k=k
        self.inp=nn.Linear(dim_in,D,bias=False)     # 1->D (轻量嵌入)
        self.enc=nn.Linear(D,N,bias=False)           # D->N
        self.out=nn.Linear(D,D,bias=False)           # D->D
        self.ln=nn.LayerNorm(D)
        self.ffn=nn.Sequential(nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D))
        self.head=nn.Linear(D,n_class)
    def forward(self,x):
        # x: (B, T, dim_in)
        B,T,_=x.size(); D=self.D; N=self.N; k=self.k
        h=self.inp(x)                  # B,T,D
        ln_h=self.ln(h)
        lat=self.enc(ln_h)             # B,T,N
        topv,topi=torch.topk(lat,k,dim=-1)
        act=torch.relu(torch.zeros_like(lat).scatter(-1,topi,topv))   # 稀疏正激活
        rho=torch.einsum('btn,btd->bnd',act,ln_h)                     # rho Hebbian 记忆
        mem=torch.einsum('btn,bnd->btd',act,rho)                      # 记忆读回
        h=h+mem
        h=h+self.ffn(self.ln(h))
        pooled=h.mean(1)               # B,D 池化
        return self.head(pooled)

def _inp(x): return torch.from_numpy(x.astype(np.float32)).reshape(x.shape[0],x.shape[1],1)  # (n,64)->(n,64,1)

def train_capfw(cap, name, seed=0, epochs=60, lr=3e-3):
    X,Y=load_skill(name,2000,seed); cap.train()
    opt=torch.optim.Adam(cap.parameters(),lr=lr); lf=nn.CrossEntropyLoss()
    Xt=_inp(X); Yt=torch.from_numpy(Y).long()
    for ep in range(epochs):
        idx=np.random.permutation(len(X))
        for i in range(0,len(X),64):
            b=idx[i:i+64]; opt.zero_grad(); l=lf(cap(Xt[b]),Yt[b]); l.backward(); opt.step()
def acc_capfw(cap, name, seed=0, n=500):
    X,Y=load_skill(name,n,seed); cap.eval()
    with torch.no_grad(): pred=cap(_inp(X)).argmax(1).numpy()
    return float((pred==Y).mean())
def cap_statefw(cap, name, seed=0):
    X,Y=load_skill(name,256,seed)
    with torch.no_grad():
        p=list(cap.parameters())
        w0=p[0].detach().cpu().numpy(); wl=p[-1].detach().cpu().numpy()
        out=cap(_inp(X)).detach().numpy()
    return np.concatenate([[w0.mean(),w0.std(),wl.mean(),wl.std()],[float(out.std())+1e-6,float(Y.std())]]).astype(np.float32)
def probe_damagefw(cap, old_names, new_name, seed=0, epochs=25):
    before={}; cap.eval()
    with torch.no_grad():
        for on in old_names:
            Xv,Yv=load_skill(on,256,seed); before[on]=float((cap(_inp(Xv)).argmax(1).numpy()==Yv).mean())
    trial=copy.deepcopy(cap); X,Y=load_skill(new_name,1200,seed); trial.train()
    opt=torch.optim.Adam(trial.parameters(),lr=3e-3); lf=nn.CrossEntropyLoss()
    Xt=_inp(X); Yt=torch.from_numpy(Y).long()
    for ep in range(epochs):
        idx=np.random.permutation(len(X))
        for i in range(0,len(X),64):
            b=idx[i:i+64]; opt.zero_grad(); l=lf(trial(Xt[b]),Yt[b]); l.backward(); opt.step()
    trial.eval(); after={}
    with torch.no_grad():
        for on in old_names:
            Xv,Yv=load_skill(on,256,seed); after[on]=float((trial(_inp(Xv)).argmax(1).numpy()==Yv).mean())
    return float(np.mean([before[o]-after[o] for o in old_names]))

# ---------- 基于 CapFW 的平衡数据集 + predictor ----------
REP={'classify':'classify_4c','extract':'extract_2c','math':'math_4c','transform':'transform_3c'}
def build_balanced_fw(seed=0, probe_epochs=25):
    rows=[]
    for cap_fam, main in REP.items():
        cap=CapFW(); train_capfw(cap,main,seed)
        cs=cap_statefw(cap,main,seed)
        for vname in VARIANTS:
            dmg=probe_damagefw(cap,[main],vname,seed,probe_epochs)
            tr=task_repv(vname,seed)
            rows.append({'feat':np.concatenate([tr,cs]).tolist(),'damage':dmg,
                         'new_name':vname,'new_family':fam_of_variant(vname),
                         'cap_family':cap_fam,'is_same_family':fam_of_variant(vname)==cap_fam})
    return rows

def train_pred(rows, seed=0, epochs=600, lr=1e-3, split_seed=0):
    feats=np.array([r['feat'] for r in rows]).astype(np.float32)
    dmg=np.array([r['damage'] for r in rows]).astype(np.float32)
    cap_fam=[r['cap_family'] for r in rows]; new_fam=[r['new_family'] for r in rows]
    is_same=np.array([r['is_same_family'] for r in rows])
    fmean,fstd=feats.mean(0),feats.std(0)+1e-6; fn=(feats-fmean)/fstd
    rng=np.random.RandomState(split_seed); pairs=sorted(set(zip(cap_fam,new_fam))); te=[]
    for p in pairs:
        cand=[i for i in range(len(rows)) if (cap_fam[i],new_fam[i])==p]
        k=max(1,int(len(cand)*0.3)); te.extend(rng.choice(cand,k,replace=False).tolist())
    tr=[i for i in range(len(rows)) if i not in te]
    m=nn.Sequential(nn.Linear(fn.shape[1],32),nn.ReLU(),nn.Linear(32,1))
    opt=torch.optim.Adam(m.parameters(),lr=lr); lf=nn.MSELoss()
    X=torch.from_numpy(fn[tr]); Y=torch.from_numpy(dmg[tr]).reshape(-1,1)
    w=torch.ones(len(tr))
    for j,i in enumerate(tr): w[j]=1.0 if is_same[i] else 0.25
    w=w/w.mean()
    for _ in range(epochs):
        opt.zero_grad(); loss=(lf(m(X),Y)*w.reshape(-1,1)).mean(); loss.backward(); opt.step()
    m.eval()
    def ev(idx):
        with torch.no_grad(): pred=m(torch.from_numpy(fn[idx])).reshape(-1).numpy()
        gt=dmg[idx]; sp=spearmanr(pred,gt); THR=0.3
        pb=(pred>=THR).astype(int); gb=(gt>=THR).astype(int)
        return sp,(pb==gb).mean(),f1_bin(gb,pb)
    return {'m':m,'fmean':fmean,'fstd':fstd}, ev(tr), ev(te)

# ---------- 池（CapFW 基底 + 免训 predictor） ----------
def predict_damage(pred, cap, rep, variant, seed=0):
    feat=np.concatenate([task_repv(variant,seed), cap_statefw(cap,rep,seed)])
    fn=((feat-pred['fmean'])/pred['fstd']).astype(np.float32)
    with torch.no_grad(): return float(pred['m'](torch.from_numpy(fn.reshape(1,-1))).item())
class PredPoolFW:
    def __init__(self,pred,thr=0.3): self.pred=pred; self.caps=[]; self.thr=thr
    def consider(self, variant, seed=0):
        fam=fam_of_variant(variant)
        if not self.caps:
            cap=CapFW(); train_capfw(cap,REP[fam],seed); self.caps.append({'model':cap,'fam':fam,'owned':{variant}})
            return {'action':'SPAWN(first)','cap':0,'damage':None}
        cand=[(ci,predict_damage(self.pred,c['model'],REP[c['fam']],variant,seed)) for ci,c in enumerate(self.caps)]
        ci,d=min(cand,key=lambda x:x[1])
        if d<self.thr:
            X,Y=load_skillv(variant,2000,seed); cap=self.caps[ci]['model']; cap.train()
            opt=torch.optim.Adam(cap.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss()
            Xt=_inp(X); Yt=torch.from_numpy(Y).long()
            for ep in range(30):
                idx=np.random.permutation(len(X))
                for i in range(0,len(X),64):
                    b=idx[i:i+64]; opt.zero_grad(); l=lf(cap(Xt[b]),Yt[b]); l.backward(); opt.step()
            self.caps[ci]['owned'].add(variant); return {'action':'REUSE','cap':ci,'damage':d}
        cap=CapFW(); train_capfw(cap,REP[fam],seed); self.caps.append({'model':cap,'fam':fam,'owned':{variant}})
        return {'action':'SPAWN','cap':len(self.caps)-1,'damage':d}

def acc_capfwv(cap, variant, seed=0, n=500):
    X,Y=load_skillv(variant,n,seed); cap.eval()
    with torch.no_grad(): pred=cap(_inp(X)).argmax(1).numpy()
    return float((pred==Y).mean())

if __name__=='__main__':
    print("=== 步骤3-续：FusedFW 作为 DCA 的 Capability substrate ===")
    print("--- ① 离训练免训 predictor（基于 CapFW cap_state，平衡数据集）---")
    rows=build_balanced_fw(seed=0); print(f"  平衡数据集 {len(rows)} 行")
    pred,ev_tr,ev_te=train_pred(rows)
    print(f"  predictor 训练(Sp={ev_tr[0]:.3f},acc={ev_tr[1]:.3f}) 留出(Sp={ev_te[0]:.3f},acc={ev_te[1]:.3f},F1={ev_te[2]:.3f})")
    print("--- ② CapFW 池跑变体序列（能力边界 + 不遗忘）---")
    seq=['classify_v0','classify_v1','extract_v0','math_v0','transform_v0']
    for seed in [0,1]:
        pool=PredPoolFW(pred); print(f"\n  [seed {seed}]")
        for v in seq:
            r=pool.consider(v,seed); d=f"{r['damage']:.3f}" if r['damage'] is not None else "N/A"
            print(f"    {v:14s} (fam={fam_of_variant(v):9s}) -> {r['action']:11s} cap#{r['cap']} 损伤={d}")
        print(f"    最终 cap 数={len(pool.caps)}")
        print(f"    分化矩阵={json.dumps({i:sorted(c['owned']) for i,c in enumerate(pool.caps)},ensure_ascii=False)}")
        for i,c in enumerate(pool.caps):
            for v in c['owned']:
                print(f"    cap#{i} [{v}] acc={acc_capfwv(c['model'],v,seed):.3f}")
