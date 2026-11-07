import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings,re; pass; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from data import load_competition
from splits import random_kfold, scaffold_kfold
from features import build
from meta_features import MetaEncoder
from title_features import build_title
tr,_=load_competition(verbose=False); y=tr.y.to_numpy()
LAB=re.compile(r"^(\d+)\s*'?\s*([a-z])?",re.I)
SUF=re.compile(r"-(as|ya|qs|nav)\b",re.I)
def ser(t):
    m=LAB.match(str(t).strip().lower()); return int(m.group(1)) if m else -1
def let(t):
    m=LAB.match(str(t).strip().lower())
    return (ord(m.group(2))-96) if (m and m.group(2)) else -1
def lab(t): return 1 if SUF.search(str(t)) else 0
def extra(d, fit=None):
    s=np.array([ser(t) for t in d.Title]); l=np.array([let(t) for t in d.Title]); b=np.array([lab(t) for t in d.Title])
    return np.c_[s,l,b,(s>=0).astype(float)]
P=dict(objective='regression',n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FP={bl:build(tr.smiles.tolist(),bl)[0] for bl in REPR}
TT,_=build_title(tr.Title.tolist()); EX=extra(tr)
sr=np.array([ser(t) for t in tr.Title])
print('номера серий в train:',sorted(set(sr[sr>=0]))[:20])
print('доля строк с распознанной серией: %.1f%%'%(100*(sr>=0).mean()))
for s in sorted(set(sr[sr>=0])):
    m=sr==s
    if m.sum()>=8: print('  серия %3d: n=%3d | медиана %7.2f | макс %9.1f | нулей %.0f%%'%(s,m.sum(),np.median(y[m]),y[m].max(),100*(y[m]==0).mean()))
ex=int(np.argmax(y))
def run(name, fn):
    out={}
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        errs=[]; errs_r=[]
        for seed in range(6):
            pred=np.zeros(len(tr))
            for trn,val in kf(tr['smiles'].tolist(),n_splits=5,seed=seed):
                pred[val]=fn(trn,val,seed)
            errs.append(np.mean((pred-y)**2))
            k=np.ones(len(tr),bool); k[ex]=False
            errs_r.append(np.mean((pred[k]-y[k])**2))
        out[sname]=(np.mean(errs),np.std(errs),np.mean(errs_r))
    return name,out
def mk(use_extra):
    def f(trn,val,seed):
        me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
        ps=[]
        for bl in REPR:
            X=FP[bl]; k=X[trn].std(0)>0
            pa=[X[trn][:,k],A,TT[trn]]; pb=[X[val][:,k],B,TT[val]]
            if use_extra: pa.append(EX[trn]); pb.append(EX[val])
            m=lgb.LGBMRegressor(random_state=seed,**P).fit(np.hstack(pa),np.log1p(y[trn]))
            ps.append(np.clip(np.expm1(m.predict(np.hstack(pb))),0,None))
        return np.mean(ps,0)
    return f
res=[run('без серии',mk(False)),run('+ серия/буква/лаб-код',mk(True))]
for sname in ('random','scaffold'):
    print('\n=== %s ==='%sname)
    for n,o in sorted(res,key=lambda r:r[1][sname][0]):
        m,sd,rb=o[sname]; print('  %-26s ПОЛНАЯ %9.0f ± %7.0f | без экстремума %9.0f'%(n,m,sd,rb))
