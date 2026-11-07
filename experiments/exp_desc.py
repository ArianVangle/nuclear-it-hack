import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings,os,pickle; pass; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from data import load_competition, canonical_smiles
from splits import random_kfold, scaffold_kfold
from features import build
from meta_features import MetaEncoder
from title_features import build_title
from binned_model import BinnedExpectationModel
from sklearn.metrics import roc_auc_score
CACHE=_CACHE + '/base.pkl'
tr,te=load_competition(verbose=False); y=tr.y.to_numpy(); S=tr.smiles.tolist()
e=pd.read_csv(_DATA + '/external.csv'); e['smiles']=e.smiles.map(canonical_smiles)
e=e[e.smiles.notna()]; e=e[~e.smiles.isin(set(te.smiles))&~e.smiles.isin(set(tr.smiles))]
a=e[(e.SI>0)&(e.SI<10000)].copy(); THR=500.
P=dict(objective='regression',n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
CL=dict(objective='binary',n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=10,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FPt={bl:build(S,bl)[0] for bl in REPR}; FPe={bl:build(a.smiles.tolist(),bl)[0] for bl in REPR}
Dt,dn=build(S,('desc',)); De,_=build(a.smiles.tolist(),('desc',))
print('дескрипторов: %d'%Dt.shape[1])
TT,_=build_title(tr.Title.tolist()); ye=(a.SI>THR).to_numpy().astype(int)
NS=4
if os.path.exists(CACHE):
    base=pickle.load(open(CACHE,'rb')); print('база из кэша')
else:
    base={}
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        for seed in range(NS):
            b=np.zeros(len(tr))
            for trn,val in kf(S,n_splits=5,seed=seed):
                me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
                LG=[]
                for bl in REPR:
                    X=FPt[bl]; k=X[trn].std(0)>0
                    m=lgb.LGBMRegressor(random_state=seed,**P).fit(np.hstack([X[trn][:,k],A,TT[trn]]),np.log1p(y[trn]))
                    LG.append(np.clip(np.expm1(m.predict(np.hstack([X[val][:,k],B,TT[val]]))),0,None))
                BN=np.mean([BinnedExpectationModel(blocks=bl,use_meta=True,use_title=True,seed=seed).fit(tr.iloc[trn]).predict(tr.iloc[val]) for bl in REPR],0)
                b[val]=0.5*np.mean(LG,0)+0.5*BN
            base[(sname,seed)]=b
    pickle.dump(base,open(CACHE,'wb')); print('база посчитана и сохранена')
# классификаторы хвоста: отпечатки / +дескрипторы / только дескрипторы
def qvar(kind,seed):
    ps=[]
    if kind=='desc':
        kk=(De.std(0)>0)&np.isfinite(De).all(0)&np.isfinite(Dt).all(0)
        return lgb.LGBMClassifier(random_state=seed,**CL).fit(De[:,kk],ye).predict_proba(Dt[:,kk])[:,1]
    for bl in REPR:
        Xe,Xt=FPe[bl],FPt[bl]
        if kind=='fp+desc':
            kk=(De.std(0)>0)&np.isfinite(De).all(0)&np.isfinite(Dt).all(0)
            Xe=np.hstack([Xe,De[:,kk]]); Xt=np.hstack([Xt,Dt[:,kk]])
        k=Xe.std(0)>0
        ps.append(lgb.LGBMClassifier(random_state=seed,**CL).fit(Xe[:,k],ye).predict_proba(Xt[:,k])[:,1])
    return np.mean(ps,0)
Q={}
for kind in ('fp','fp+desc','desc'):
    Q[kind]=np.mean([qvar(kind,s) for s in range(3)],0)
    print('  %-8s AUC(SI>500) на train %.4f | AUC(SI>1000) %.4f'%(kind,roc_auc_score((y>500).astype(int),Q[kind]),roc_auc_score((y>1000).astype(int),Q[kind])))
Q['fp+desc смесь']=0.5*(Q['fp']+Q['desc'])
print('  %-8s AUC(SI>500) %.4f | AUC(SI>1000) %.4f'%('смесь',roc_auc_score((y>500).astype(int),Q['fp+desc смесь']),roc_auc_score((y>1000).astype(int),Q['fp+desc смесь'])))
E_TAIL=float(np.percentile(a.SI[a.SI>THR],75))
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
print('\n%-22s %10s %10s %10s | %10s'%('','УСТ rnd','УСТ scf','УСТ смесь','ПОЛН смесь'))
rows=[]
for kind,q in Q.items():
    o={}
    for sname in ('random','scaffold'):
        R=[];F=[]
        for seed in range(NS):
            b=base[(sname,seed)]; qq=q**2; p=qq*E_TAIL+(1-qq)*b
            R.append(np.mean((p[keep]-y[keep])**2)); F.append(np.mean((p-y)**2))
        o[sname]=(np.mean(R),np.mean(F))
    rows.append((kind,o['random'][0],o['scaffold'][0],0.668*o['random'][0]+0.332*o['scaffold'][0],0.668*o['random'][1]+0.332*o['scaffold'][1]))
for r in sorted(rows,key=lambda x:x[3]): print('%-22s %10.0f %10.0f %10.0f | %10.0f'%r)
