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
tr,te=load_competition(verbose=False); y=tr.y.to_numpy(); S=tr.smiles.tolist()
e=pd.read_csv(_DATA + '/external.csv'); e['smiles']=e.smiles.map(canonical_smiles)
e=e[e.smiles.notna()]; e=e[~e.smiles.isin(set(te.smiles))&~e.smiles.isin(set(tr.smiles))]
a=e[(e.SI>0)&(e.SI<10000)].copy(); THR=500.; E_TAIL=float(np.percentile(a.SI[a.SI>THR],75))
P=dict(objective='regression',n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
CL=dict(objective='binary',n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=10,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FPt={bl:build(S,bl)[0] for bl in REPR}; FPe={bl:build(a.smiles.tolist(),bl)[0] for bl in REPR}
Dt,_=build(S,('desc',)); De,_=build(a.smiles.tolist(),('desc',))
kd=(De.std(0)>0)&(Dt.std(0)>0)&np.isfinite(De).all(0)&np.isfinite(Dt).all(0)
Dt=Dt[:,kd]; De=De[:,kd]
TT,_=build_title(tr.Title.tolist()); ye=(a.SI>THR).to_numpy().astype(int)
# хвост: отпечатки+дескрипторы (принятый вариант)
def qtail(seed):
    ps=[]
    for bl in REPR:
        Xe=np.hstack([FPe[bl],De]); Xt=np.hstack([FPt[bl],Dt]); k=Xe.std(0)>0
        ps.append(lgb.LGBMClassifier(random_state=seed,**CL).fit(Xe[:,k],ye).predict_proba(Xt[:,k])[:,1])
    return np.mean(ps,0)
QT=np.mean([qtail(s) for s in range(3)],0)**2
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
NS=4
def run(use_desc):
    o={}
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        R=[];F=[]
        for seed in range(NS):
            b=np.zeros(len(tr))
            for trn,val in kf(S,n_splits=5,seed=seed):
                me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
                LG=[]
                for bl in REPR:
                    X=FPt[bl]
                    Xa=np.hstack([X[trn],Dt[trn]]) if use_desc else X[trn]
                    Xb=np.hstack([X[val],Dt[val]]) if use_desc else X[val]
                    k=Xa.std(0)>0
                    m=lgb.LGBMRegressor(random_state=seed,**P).fit(np.hstack([Xa[:,k],A,TT[trn]]),np.log1p(y[trn]))
                    LG.append(np.clip(np.expm1(m.predict(np.hstack([Xb[:,k],B,TT[val]]))),0,None))
                bl2=[tuple(list(bl)+['desc']) if use_desc else bl for bl in REPR]
                BN=np.mean([BinnedExpectationModel(blocks=bb,use_meta=True,use_title=True,seed=seed).fit(tr.iloc[trn]).predict(tr.iloc[val]) for bb in bl2],0)
                b[val]=0.5*np.mean(LG,0)+0.5*BN
            p=QT*E_TAIL+(1-QT)*b
            R.append(np.mean((p[keep]-y[keep])**2)); F.append(np.mean((p-y)**2))
        o[sname]=(np.mean(R),np.std(R),np.mean(F))
    return o
for nm,ud in (('основная без desc',False),('основная + desc',True)):
    o=run(ud)
    print('%-20s УСТ rnd %8.0f ± %6.0f | УСТ scf %8.0f | УСТ смесь %8.0f | ПОЛН смесь %9.0f'%(
        nm,o['random'][0],o['random'][1],o['scaffold'][0],
        0.668*o['random'][0]+0.332*o['scaffold'][0],
        0.668*o['random'][2]+0.332*o['scaffold'][2]))
