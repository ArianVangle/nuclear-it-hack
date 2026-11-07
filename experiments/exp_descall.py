import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings; pass; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from data import load_competition, canonical_smiles
from splits import random_kfold, scaffold_kfold
from features import build
from meta_features import MetaEncoder
from title_features import build_title
from binned_model import BinnedExpectationModel
from sklearn.metrics import roc_auc_score
tr,te=load_competition(verbose=False); y=tr.y.to_numpy(); S=tr.smiles.tolist()
e=pd.read_csv(_DATA + '/external.csv'); e['smiles']=e.smiles.map(canonical_smiles)
e=e[e.smiles.notna()]; e=e[~e.smiles.isin(set(te.smiles))&~e.smiles.isin(set(tr.smiles))]
a=e[(e.SI>0)&(e.SI<10000)].copy(); THR=500.
E_TAIL=float(np.percentile(a.SI[a.SI>THR],75)); ye=(a.SI>THR).to_numpy().astype(int)
P=dict(objective='regression',n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
CL=dict(objective='binary',n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=10,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FPt={bl:build(S,bl)[0] for bl in REPR}; FPe={bl:build(a.smiles.tolist(),bl)[0] for bl in REPR}
D={}
for tag in ('desc','descall'):
    At,_=build(S,(tag,)); Ae,_=build(a.smiles.tolist(),(tag,))
    k=(np.isfinite(At).all(0))&(np.isfinite(Ae).all(0))&(At.std(0)>0)&(Ae.std(0)>0)
    D[tag]=(At[:,k],Ae[:,k]); print('%s: %d пригодных'%(tag,k.sum()))
TT,_=build_title(tr.Title.tolist())
Q={}
for tag in ('desc','descall'):
    Dt,De=D[tag]; ps=[]
    for bl in REPR:
        Xe=np.hstack([FPe[bl],De]); Xt=np.hstack([FPt[bl],Dt]); k=Xe.std(0)>0
        for s in range(3): ps.append(lgb.LGBMClassifier(random_state=s,**CL).fit(Xe[:,k],ye).predict_proba(Xt[:,k])[:,1])
    Q[tag]=np.mean(ps,0)**2
    print('хвост %-8s AUC(y>500) %.4f | AUC(y>1000) %.4f'%(tag,roc_auc_score((y>500).astype(int),Q[tag]),roc_auc_score((y>1000).astype(int),Q[tag])))
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
NS=3
BASE={}
for tag in ('desc','descall'):
    Dt,_=D[tag]
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        for seed in range(NS):
            b=np.zeros(len(tr))
            for trn,val in kf(S,n_splits=5,seed=seed):
                me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
                LG=[]
                for bl in REPR:
                    X=FPt[bl]
                    Xa=np.hstack([X[trn],Dt[trn]]); Xb=np.hstack([X[val],Dt[val]]); k=Xa.std(0)>0
                    m=lgb.LGBMRegressor(random_state=seed,**P).fit(np.hstack([Xa[:,k],A,TT[trn]]),np.log1p(y[trn]))
                    LG.append(np.clip(np.expm1(m.predict(np.hstack([Xb[:,k],B,TT[val]]))),0,None))
                BN=np.mean([BinnedExpectationModel(blocks=tuple(list(bl)+[tag]),use_meta=True,use_title=True,seed=seed).fit(tr.iloc[trn]).predict(tr.iloc[val]) for bl in REPR],0)
                b[val]=0.5*np.mean(LG,0)+0.5*BN
            BASE[(tag,sname,seed)]=b
    print('база посчитана:',tag)
print('\n%-32s %10s %10s %10s | %10s'%('основная / хвост','УСТ rnd','УСТ scf','УСТ смесь','ПОЛН смесь'))
rows=[]
for mt in ('desc','descall'):
    for qt in ('desc','descall'):
        o={}
        for sname in ('random','scaffold'):
            R=[];F=[]
            for seed in range(NS):
                b=BASE[(mt,sname,seed)]; q=Q[qt]; p=q*E_TAIL+(1-q)*b
                R.append(np.mean((p[keep]-y[keep])**2)); F.append(np.mean((p-y)**2))
            o[sname]=(np.mean(R),np.mean(F))
        rows.append(('%s / %s'%(mt,qt),o['random'][0],o['scaffold'][0],
                     0.668*o['random'][0]+0.332*o['scaffold'][0],
                     0.668*o['random'][1]+0.332*o['scaffold'][1]))
for r in sorted(rows,key=lambda x:x[3]): print('%-32s %10.0f %10.0f %10.0f | %10.0f'%r)
