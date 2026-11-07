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
a=e[(e.SI>0)&(e.SI<10000)].copy()
THR=500.; E_TAIL=float(np.percentile(a.SI[a.SI>THR],75))
print('E_TAIL=%.0f, внешних положительных %d'%(E_TAIL,(a.SI>THR).sum()))
P=dict(objective='regression',n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
CL=dict(objective='binary',n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=10,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FPt={bl:build(S,bl)[0] for bl in REPR}; FPe={bl:build(a.smiles.tolist(),bl)[0] for bl in REPR}
TT,_=build_title(tr.Title.tolist()); ye=(a.SI>THR).to_numpy().astype(int); yt=(y>THR).astype(int)
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
NS=4
store={}
for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
    for seed in range(NS):
        base=np.zeros(len(tr)); qe=np.zeros(len(tr)); qb=np.zeros(len(tr))
        for trn,val in kf(S,n_splits=5,seed=seed):
            me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
            LG=[];QE=[];QB=[]
            for bl in REPR:
                X=FPt[bl]; k=X[trn].std(0)>0
                m=lgb.LGBMRegressor(random_state=seed,**P).fit(np.hstack([X[trn][:,k],A,TT[trn]]),np.log1p(y[trn]))
                LG.append(np.clip(np.expm1(m.predict(np.hstack([X[val][:,k],B,TT[val]]))),0,None))
                Xe=FPe[bl]; ke=Xe.std(0)>0
                QE.append(lgb.LGBMClassifier(random_state=seed,**CL).fit(Xe[:,ke],ye).predict_proba(X[val][:,ke])[:,1])
                Xc=np.vstack([Xe,X[trn]]); yy=np.r_[ye,yt[trn]]; kc=Xc.std(0)>0
                QB.append(lgb.LGBMClassifier(random_state=seed,**CL).fit(Xc[:,kc],yy).predict_proba(X[val][:,kc])[:,1])
            BN=np.mean([BinnedExpectationModel(blocks=bl,use_meta=True,use_title=True,seed=seed).fit(tr.iloc[trn]).predict(tr.iloc[val]) for bl in REPR],0)
            base[val]=0.5*np.mean(LG,0)+0.5*BN; qe[val]=np.mean(QE,0); qb[val]=np.mean(QB,0)
        store[(sname,seed)]=(base,qe,qb)
    print('посчитано:',sname)
VAR={'A без хвоста':lambda b,e_,b_: b,
     'B сырая q (внешняя)':lambda b,e_,b_: e_*E_TAIL+(1-e_)*b,
     'C q^2 (внешняя)':lambda b,e_,b_: e_**2*E_TAIL+(1-e_**2)*b,
     'D сырая q (внешн+сорев)':lambda b,e_,b_: b_*E_TAIL+(1-b_)*b,
     'E q^2 (внешн+сорев)':lambda b,e_,b_: b_**2*E_TAIL+(1-b_**2)*b}
res={}
for n,f in VAR.items():
    row={}
    for sname in ('random','scaffold'):
        F=[];R=[]
        for seed in range(NS):
            b,e_,b_=store[(sname,seed)]; p=f(b,e_,b_)
            F.append(np.mean((p-y)**2)); R.append(np.mean((p[keep]-y[keep])**2))
        row[sname]=(np.mean(F),np.std(F),np.mean(R))
    row['смесь']=(0.668*row['random'][0]+0.332*row['scaffold'][0],
                  0.668*row['random'][2]+0.332*row['scaffold'][2])
    res[n]=row
print('\n%-26s %10s %10s %10s | %10s %10s %10s'%('','ПОЛН rnd','ПОЛН scf','ПОЛН смесь','УСТ rnd','УСТ scf','УСТ смесь'))
for n,r in sorted(res.items(),key=lambda x:x[1]['смесь'][0]):
    print('%-26s %10.0f %10.0f %10.0f | %10.0f %10.0f %10.0f'%(n,r['random'][0],r['scaffold'][0],r['смесь'][0],r['random'][2],r['scaffold'][2],r['смесь'][1]))
print('\n(УСТ = без молекулы 36067; смесь = 0.668*random + 0.332*scaffold под состав теста)')
