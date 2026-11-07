import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings,pickle; pass; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from data import load_competition, canonical_smiles
from features import build
from sklearn.metrics import roc_auc_score
tr,te=load_competition(verbose=False); y=tr.y.to_numpy()
base=pickle.load(open(_CACHE + '/base.pkl','rb'))
e=pd.read_csv(_DATA + '/external.csv'); e['cs']=e.smiles.map(canonical_smiles)
e=e[e.cs.notna()&~e.cs.isin(set(te.smiles))&~e.cs.isin(set(tr.smiles))].copy()
e['cc']=pd.to_numeric(e.CC50,errors='coerce'); e['ic']=pd.to_numeric(e.IC50,errors='coerce')
rel=(e.cc/e.ic)/e.SI
CUR=e[(e.SI>0)&(e.SI<10000)].copy()
CLEAN=e[(e.SI>0)&(e.SI!=10000)&(rel>0.9)&(rel<1.1)].copy()
print('текущее %d (полож>500 %d) | чистое %d (полож>500 %d)'%(len(CUR),(CUR.SI>500).sum(),len(CLEAN),(CLEAN.SI>500).sum()))
CL=dict(objective='binary',n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=10,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
Dt,_=build(tr.smiles.tolist(),('desc',))
FPt={bl:build(tr.smiles.tolist(),bl)[0] for bl in REPR}
def qmodel(src,thr):
    De,_=build(src.cs.tolist(),('desc',))
    kd=(De.std(0)>0)&(Dt.std(0)>0)&np.isfinite(De).all(0)&np.isfinite(Dt).all(0)
    yy=(src.SI>thr).to_numpy().astype(int)
    if yy.sum()<8: return None
    ps=[]
    for bl in REPR:
        Xe=np.hstack([build(src.cs.tolist(),bl)[0],De[:,kd]]); Xt=np.hstack([FPt[bl],Dt[:,kd]])
        k=Xe.std(0)>0
        for s in range(3): ps.append(lgb.LGBMClassifier(random_state=s,**CL).fit(Xe[:,k],yy).predict_proba(Xt[:,k])[:,1])
    return np.mean(ps,0)
THRS=[500,1000,2000,5000]
Q={}
for nm,src in (('текущее',CUR),('чистое',CLEAN)):
    Q[nm]={t:qmodel(src,t) for t in THRS}
    for t in THRS:
        q=Q[nm][t]
        if q is None: print('  %-8s порог %5d: мало положительных'%(nm,t)); continue
        print('  %-8s порог %5d (полож %3d) | AUC(y>500) %.4f | AUC(y>1000) %.4f'%(nm,t,(src.SI>t).sum(),roc_auc_score((y>500).astype(int),q),roc_auc_score((y>1000).astype(int),q)))
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
def ev(predfn):
    o={}
    for sname in ('random','scaffold'):
        R=[];F=[]
        for seed in range(4):
            p=predfn(base[(sname,seed)])
            R.append(np.mean((p[keep]-y[keep])**2)); F.append(np.mean((p-y)**2))
        o[sname]=(np.mean(R),np.mean(F))
    return (0.668*o['random'][0]+0.332*o['scaffold'][0], 0.668*o['random'][1]+0.332*o['scaffold'][1], o['random'][0], o['scaffold'][0])
# представители интервалов по внешнему распределению
def reps(src):
    S=src.SI.to_numpy(); out={}
    edges=THRS+[1e9]
    for i,t in enumerate(THRS):
        m=(S>t)&(S<=edges[i+1])
        out[t]=float(np.median(S[m])) if m.sum()>3 else float(t*1.8)
    return out
rows=[]
for nm,src in (('текущее',CUR),('чистое',CLEAN)):
    E_T=float(np.percentile(src.SI[src.SI>500],75))
    q5=Q[nm][500]
    rows.append(('%s: один хвост (q^2)'%nm, ev(lambda b,q=q5,E=E_T: (q**2)*E+(1-q**2)*b)))
    RP=reps(src)
    def stair(b,Qs=Q[nm],RP=RP):
        p=b.copy(); prev=np.zeros(len(b)); add=np.zeros(len(b)); rem=np.ones(len(b))
        # P(в интервале i) = q_i - q_{i+1}, сжатие как q^2
        qq={t:(Qs[t]**2 if Qs[t] is not None else np.zeros(len(b))) for t in THRS}
        for i,t in enumerate(THRS):
            nxt=qq[THRS[i+1]] if i+1<len(THRS) else np.zeros(len(b))
            w=np.clip(qq[t]-nxt,0,None)
            add+=w*RP[t]; rem-=w
        return np.clip(rem,0,1)*b+add
    rows.append(('%s: ступенчато (q^2)'%nm, ev(stair)))
print('\n%-28s %10s %10s %10s | %10s'%('','УСТ rnd','УСТ scf','УСТ смесь','ПОЛН смесь'))
for n,(mix,fmix,r,s) in sorted(rows,key=lambda x:x[1][0]):
    print('%-28s %10.0f %10.0f %10.0f | %10.0f'%(n,r,s,mix,fmix))
