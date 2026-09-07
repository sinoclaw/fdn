"""步骤3：接 DCA 免训 Interference Predictor 进动态能力池（PredictorDynPool）。

DCA 池现有 DynPool.consider() 用"真 probe"（深拷贝 cap + 训练 new_skill）测损伤 → 成本随 cap 规模涨。
免训 Predictor（MLP: [task_rep; cap_state] → 预测 damage，29.1µs 规模无关）把决策改成纯前向，零训练。

本文件：
  1. 复用 dca_v1_ip2 的统一技能集（classify_4c/3c, extract_2c, math_4c, transform_3c，全统一 4 类）
  2. 离线构建 (feat=[task_rep;cap_state] → damage) 数据集 + 训练 predictor（一次，验证 Spearman/一致/F1）
  3. PredictorDynPool.consider()：候选 cap 用 predictor 预测损伤 → 低则 REUSE、高则 SPAWN（免训，无真 probe）
  4. 跑技能序列，验证：能力分化矩阵正确（同族复查复用/异族新建）+ 旧能力不遗忘 + REUSE/SPAWN 决策 vs 真probe 一致
"""
import numpy as np, torch, torch.nn as nn, sys, json
sys.path.insert(0,'/tmp')
from dca_v1_ip2 import (SKILLS, load_skill, task_rep_skill, Cap, train_cap, acc_cap,
                        cap_state, probe_damage, N_CLASS, build_dataset, spearmanr, f1_bin)

def rankdata(x):
    order=np.argsort(x,kind='mergesort'); r=np.empty(len(x),float); i=0
    while i<len(x):
        j=i
        while j+1<len(x) and x[order[j+1]]==x[order[i]]: j+=1
        avg=(i+j)/2.0+1.0
        for k in range(i,j+1): r[order[k]]=avg
        i=j+1
    return r

# ---------- 离线：训练免训 predictor ----------
def train_predictor(seed=0, probe_epochs=25, epochs=600, lr=1e-3, split_seed=0):
    rows = build_dataset(seed=seed, probe_epochs=probe_epochs)
    feats = np.array([r['feat'] for r in rows]).astype(np.float32)
    dmg = np.array([r['damage'] for r in rows]).astype(np.float32)
    is_same = np.array([r['is_same_family'] for r in rows])
    cap_fam=[r['cap_family'] for r in rows]; new_fam=[r['new_family'] for r in rows]
    fmean,fstd = feats.mean(0), feats.std(0)+1e-6
    feats_n = (feats-fmean)/fstd
    # 留出 split（按 (cap_fam,new_fam) 对）
    rng=np.random.RandomState(split_seed); pairs=sorted(set(zip(cap_fam,new_fam))); test=[]
    for p in pairs:
        cand=[i for i in range(len(rows)) if (cap_fam[i],new_fam[i])==p]
        k=max(1,int(len(cand)*0.3)); test.extend(rng.choice(cand,k,replace=False).tolist())
    tr=[i for i in range(len(rows)) if i not in test]; te=test
    m=nn.Sequential(nn.Linear(feats_n.shape[1],32),nn.ReLU(),nn.Linear(32,1))
    opt=torch.optim.Adam(m.parameters(),lr=lr); lf=nn.MSELoss()
    X=torch.from_numpy(feats_n[tr]); Y=torch.from_numpy(dmg[tr]).reshape(-1,1)
    w=torch.ones(len(tr))
    for j,i in enumerate(tr): w[j]=1.0 if is_same[i] else 0.25
    w=w/w.mean()
    for _ in range(epochs):
        opt.zero_grad(); loss=(lf(m(X),Y)*w.reshape(-1,1)).mean(); loss.backward(); opt.step()
    m.eval()
    def ev(idx):
        with torch.no_grad(): pred=m(torch.from_numpy(feats_n[idx])).reshape(-1).numpy()
        gt=dmg[idx]; sp=spearmanr(pred,gt); THR=0.3
        pb=(pred>=THR).astype(int); gb=(gt>=THR).astype(int)
        return sp,(pb==gb).mean(),f1_bin(gb,pb),pred
    sp_tr,acc_tr,f1_tr,_=ev(tr); sp,acc,f1,pred=ev(te)
    print(f"  [predictor] 训练(Sp={sp_tr:.3f},acc={acc_tr:.3f}) 留出(Sp={sp:.3f},acc={acc:.3f},F1={f1:.3f}) n={len(rows)}")
    return {'m':m,'fmean':fmean,'fstd':fstd,'rows':rows,'sp':sp,'acc':acc,'f1':f1,'is_same':is_same}

# ---------- 免训 PredictorDynPool ----------
class PredictorDynPool:
    """用免训 predictor 替代真 probe 做 REUSE/SPAWN 决策。"""
    def __init__(self, pred):
        self.pred=pred; self.caps=[]; self.thr=0.3
    def predict_damage(self, cap, new_name, seed=0):
        with torch.no_grad():
            feats = np.concatenate([task_rep_skill(new_name,seed), cap_state(cap, new_name, seed)])
            fn = ((feats-self.pred['fmean'])/self.pred['fstd']).astype(np.float32)
            return float(self.pred['m'](torch.from_numpy(fn.reshape(1,-1))).item())
    def consider(self, new_name, seed=0):
        if not self.caps:
            cap=Cap(N_CLASS); train_cap(cap,new_name,seed)
            self.caps.append({'model':cap,'owned':{new_name}})
            return {'action':'SPAWN(first)','cap':0,'damage':None}
        cand=[]
        for ci,c in enumerate(self.caps):
            d=self.predict_damage(c['model'],new_name,seed)   # 免训预测
            cand.append((ci,d))
        ci,d=min(cand,key=lambda x:x[1])
        if d<self.thr:
            train_cap(self.caps[ci]['model'],new_name,seed,epochs=40,lr=1e-3)
            self.caps[ci]['owned'].add(new_name)
            return {'action':'REUSE','cap':ci,'damage':d}
        cap=Cap(N_CLASS); train_cap(cap,new_name,seed)
        self.caps.append({'model':cap,'owned':{new_name}})
        return {'action':'SPAWN','cap':len(self.caps)-1,'damage':d}

# ---------- 决策对账：免训 predictor vs 真 probe ----------
def check_decision_consistency(pred, seed=0):
    """对每个 (cap_skill, new_skill) 组合，比较 predictor 预测 vs 真 probe 的 REUSE/SPAWN 判断。"""
    names=list(SKILLS.keys()); rows_cmp=[]
    for main in names:
        cap=Cap(N_CLASS); train_cap(cap,main,seed)
        for nw in names:
            if nw==main: continue
            p=pred['m'](torch.from_numpy((((np.concatenate([task_rep_skill(nw,seed),cap_state(cap,nw,seed)])-pred['fmean'])/pred['fstd']).astype(np.float32)).reshape(1,-1))).item()
            real=probe_damage(cap,[main],nw,seed)
            rows_cmp.append({'main':main,'new':nw,'pred':p,'real':real,
                             'same':SKILLS[nw]['fam']==SKILLS[main]['fam']})
    real=np.array([r['real'] for r in rows_cmp]); pred=np.array([r['pred'] for r in rows_cmp])
    same=np.array([1.0 if r['same'] else 0.0 for r in rows_cmp])
    vs=spearmanr(pred,real)
    # REUSE/SPAWN 判定一致（真probe 阈值 0.15，predictor 阈值 0.3，各取最优阈值对齐）
    def thr_best(vals,gs):
        best=(0,0)
        for t in np.arange(0.0,0.6,0.01):
            a=((vals>=t)==(gs>=0.15)).mean()
            if a>best[0]: best=(a,t)
        return best
    a,t=thr_best(pred,real)
    print(f"  [对账] predictor vs 真probe damage：Spearman={vs:.3f} | REUSE/SPAWN判定一致={a:.3f}(最佳阈值{t:.2f}) | n={len(rows_cmp)}")
    return vs,a

if __name__=='__main__':
    print("=== 步骤3：接 DCA 免训 Interference Predictor 进动态能力池 ===")
    print("--- ① 离线训练免训 predictor ---")
    pred=train_predictor(seed=0)
    print("--- ② 免训 predictor vs 真probe 对账 ---")
    check_decision_consistency(pred, seed=0)
    print("--- ③ PredictorDynPool 跑技能序列（能力边界发现，判据 A）---")
    seq=['classify_4c','classify_3c','extract_2c','math_4c','transform_3c']
    for seed in [0,1]:
        pool=PredictorDynPool(pred)
        print(f"\n  [seed {seed}]")
        for sk in seq:
            r=pool.consider(sk,seed)
            d=f"{r['damage']:.3f}" if r['damage'] is not None else "N/A"
            print(f"    {sk:14s} (fam={SKILLS[sk]['fam']:9s}) -> {r['action']:11s} cap#{r['cap']} 损伤={d}")
        print(f"    最终 cap 数 = {len(pool.caps)}")
        owned={i:sorted(c['owned']) for i,c in enumerate(pool.caps)}
        print(f"    分化矩阵 = {json.dumps(owned,ensure_ascii=False)}")
        for i,c in enumerate(pool.caps):
            for os_ in c['owned']:
                print(f"    cap#{i} [{os_}] acc={acc_cap(c['model'],os_,seed):.3f}")
