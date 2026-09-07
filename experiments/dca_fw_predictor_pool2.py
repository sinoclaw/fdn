"""步骤3 v2：用 ip3 平衡多变体数据集训练"能泛化"的免训 predictor，再进 PredictorDynPool。

v1 教训：用 ip2 小数据集(16行/1同族对) → predictor 过拟合(留出 Sp=-0.29) → Pool 决策全错 → 灾难性遗忘。
v2 修正：复用 ip3 的 make_many_variants/build_balanced（每族多变体、REUSE/SPAWN 平衡）训练 predictor，
        再用 predictor 驱动池决策，验证能力分化矩阵正确 + 旧能力不遗忘。
"""
import numpy as np, torch, torch.nn as nn, sys, json
sys.path.insert(0,'/tmp')
from dca_v1_ip2 import Cap, train_cap, acc_cap, cap_state, SKILLS, N_CLASS
from dca_v1_ip3 import (make_many_variants, load_skillv, task_repv, probe_damagev,
                        VARIANTS, fam_of_variant, build_balanced, rankdata, spearmanr, f1_bin)

# 每族的代表 base 技能（cap 用它训练，cap_state 也对照它）
REP={'classify':'classify_4c','extract':'extract_2c','math':'math_4c','transform':'transform_3c'}

def train_pred_on_rows(rows, seed=0, epochs=600, lr=1e-3, split_seed=0):
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
    sp_tr,ac_tr,f1_tr=ev(tr); sp,ac,f1=ev(te)
    print(f"  [predictor] n={len(rows)} 训练(Sp={sp_tr:.3f},acc={ac_tr:.3f}) 留出(Sp={sp:.3f},acc={ac:.3f},F1={f1:.3f})")
    return {'m':m,'fmean':fmean,'fstd':fstd}

def cap_statev(cap, rep_base, seed=0):
    """cap_state 对照按 family 的代表 base 技能（与 build_balanced 一致）。"""
    from dca_v1_ip2 import load_skill as ls
    X,Y=ls(rep_base,256,seed)
    with torch.no_grad():
        w0=list(cap.net.parameters())[0].detach().cpu().numpy()
        wl=list(cap.net.parameters())[-1].detach().cpu().numpy()
        out=cap(torch.from_numpy(X.astype(np.float32))).detach().numpy()
    return np.concatenate([[w0.mean(),w0.std(),wl.mean(),wl.std()],[float(out.std())+1e-6,float(Y.std())]]).astype(np.float32)

def predict_damage(pred, cap, rep_base, variant, seed=0):
    feat=np.concatenate([task_repv(variant,seed), cap_statev(cap, rep_base, seed)])
    fn=((feat-pred['fmean'])/pred['fstd']).astype(np.float32)
    with torch.no_grad():
        return float(pred['m'](torch.from_numpy(fn.reshape(1,-1))).item())

class PredPool:
    def __init__(self,pred,thr=0.3):
        self.pred=pred; self.caps=[]; self.thr=thr
    def consider(self, variant, seed=0):
        fam=fam_of_variant(variant)
        if not self.caps:
            cap=Cap(N_CLASS); train_cap(cap,REP[fam],seed); # 用代表base训练首cap
            self.caps.append({'model':cap,'fam':fam,'owned':{variant}})
            return {'action':'SPAWN(first)','cap':0,'damage':None}
        cand=[]
        for ci,c in enumerate(self.caps):
            d=predict_damage(self.pred,c['model'],REP[c['fam']],variant,seed)
            cand.append((ci,d))
        ci,d=min(cand,key=lambda x:x[1])
        # 复用：把 variant 训练到该 cap（低lr短训）
        if d<self.thr:
            from dca_v1_ip3 import load_skillv
            X,Y=load_skillv(variant,2000,seed); cap=self.caps[ci]['model']; cap.train()
            opt=torch.optim.Adam(cap.parameters(),lr=1e-3); lf=nn.CrossEntropyLoss()
            Xt=torch.from_numpy(X.astype(np.float32)); Yt=torch.from_numpy(Y).long()
            for ep in range(30):
                idx=np.random.permutation(len(X))
                for i in range(0,len(X),64):
                    b=idx[i:i+64]; opt.zero_grad(); l=lf(cap(Xt[b]),Yt[b]); l.backward(); opt.step()
            self.caps[ci]['owned'].add(variant)
            return {'action':'REUSE','cap':ci,'damage':d}
        cap=Cap(N_CLASS); train_cap(cap,REP[fam],seed)
        self.caps.append({'model':cap,'fam':fam,'owned':{variant}})
        return {'action':'SPAWN','cap':len(self.caps)-1,'damage':d}

def acc_capv(cap, variant, seed=0, n=500):
    X,Y=load_skillv(variant,n,seed); cap.eval()
    with torch.no_grad(): pred=cap(torch.from_numpy(X.astype(np.float32))).argmax(1).numpy()
    return float((pred==Y).mean())

if __name__=='__main__':
    print("=== 步骤3 v2：平衡数据集免训 predictor 进动态能力池 ===")
    # 训练 predictor（用 ip3 平衡数据集的两族代表 cap，对多变体）
    rows=build_balanced(seed=0)
    print(f"  平衡数据集 {len(rows)} 行")
    pred=train_pred_on_rows(rows)
    # 跑一个"同族应复用/异族应spawn"的变体序列
    seq=['classify_v0','classify_v1','extract_v0','math_v0','transform_v0']
    for seed in [0,1]:
        pool=PredPool(pred); print(f"\n  [seed {seed}]")
        for v in seq:
            r=pool.consider(v,seed)
            d=f"{r['damage']:.3f}" if r['damage'] is not None else "N/A"
            print(f"    {v:14s} (fam={fam_of_variant(v):9s}) -> {r['action']:11s} cap#{r['cap']} 损伤={d}")
        print(f"    最终 cap 数={len(pool.caps)}")
        owned={i:sorted(c['owned']) for i,c in enumerate(pool.caps)}
        print(f"    分化矩阵={json.dumps(owned,ensure_ascii=False)}")
        for i,c in enumerate(pool.caps):
            for v in c['owned']:
                print(f"    cap#{i} [{v}] acc={acc_capv(c['model'],v,seed):.3f}")
