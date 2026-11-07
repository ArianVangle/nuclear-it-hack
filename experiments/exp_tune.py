import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings,itertools; pass; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from data import load_competition
from splits import random_kfold, scaffold_kfold
from features import build
from meta_features import MetaEncoder
from title_features import build_title
from final_model_v2 import literature_tables, rebuilt_tables, merge_tables, flat_smiles, _lit_features
tr,te=load_competition(verbose=False); y=tr.y.to_numpy(); S=tr.smiles.tolist()
tr['flat']=tr.smiles.map(flat_smiles)
T=merge_tables(literature_tables(_DATA + '/raw/chembl_activities.csv'),
               rebuilt_tables(_DATA + '/external_rebuilt.csv'))
LIT=_lit_features(tr,T)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FP={bl:build(S,bl)[0] for bl in REPR}
D,_=build(S,('descall',)); kd=(np.isfinite(D).all(0))&(D.std(0)>0); D=D[:,kd]
TT,_=build_title(tr.Title.tolist())
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
BASE=dict(objective='regression',verbose=-1,subsample=0.8,subsample_freq=1)
GRID=[
 dict(n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,colsample_bytree=0.8,reg_lambda=5.),   # текущая
 dict(n_estimators=400,learning_rate=0.05,num_leaves=15,min_child_samples=10,colsample_bytree=0.8,reg_lambda=5.),
 dict(n_estimators=400,learning_rate=0.05,num_leaves=4,min_child_samples=10,colsample_bytree=0.8,reg_lambda=5.),
 dict(n_estimators=800,learning_rate=0.03,num_leaves=7,min_child_samples=10,colsample_bytree=0.8,reg_lambda=5.),
 dict(n_estimators=250,learning_rate=0.08,num_leaves=7,min_child_samples=10,colsample_bytree=0.8,reg_lambda=5.),
 dict(n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=5,colsample_bytree=0.8,reg_lambda=5.),
 dict(n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=20,colsample_bytree=0.8,reg_lambda=5.),
 dict(n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,colsample_bytree=0.4,reg_lambda=5.),
 dict(n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,colsample_bytree=0.8,reg_lambda=20.),
 dict(n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,colsample_bytree=0.8,reg_lambda=1.),
 dict(n_estimators=600,learning_rate=0.04,num_leaves=5,min_child_samples=15,colsample_bytree=0.6,reg_lambda=10.),
 dict(n_estimators=400,learning_rate=0.05,num_leaves=31,min_child_samples=10,colsample_bytree=0.8,reg_lambda=5.),
]
def run(prm):
    o={}
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        R=[]
        for seed in range(2):
            b=np.zeros(len(tr))
            for trn,val in kf(S,n_splits=5,seed=seed):
                me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
                ps=[]
                for bl in REPR:
                    Xa=np.hstack([FP[bl][trn],D[trn],A,TT[trn],LIT[trn]])
                    Xb=np.hstack([FP[bl][val],D[val],B,TT[val],LIT[val]])
                    m=lgb.LGBMRegressor(random_state=seed,**BASE,**prm).fit(Xa,np.log1p(y[trn]))
                    ps.append(np.clip(np.expm1(m.predict(Xb)),0,None))
                b[val]=np.mean(ps,0)
            R.append(np.mean((b[keep]-y[keep])**2))
        o[sname]=(float(np.mean(R)),float(np.std(R)))
    return o
rows=[]
for i,p in enumerate(GRID):
    o=run(p)
    mix=0.668*o['random'][0]+0.332*o['scaffold'][0]
    rows.append((i,p,o['random'][0],o['random'][1],o['scaffold'][0],mix))
    print('%2d/%d  листьев %2d lr %.2f деревьев %3d minchild %2d colsample %.1f lambda %4.1f -> смесь %9.0f'%(
        i+1,len(GRID),p['num_leaves'],p['learning_rate'],p['n_estimators'],p['min_child_samples'],
        p['colsample_bytree'],p['reg_lambda'],mix),flush=True)
print('\n%-54s %11s %9s %11s %11s'%('конфигурация','УСТ rnd','±','УСТ scf','смесь'))
for i,p,r,sd,s,mix in sorted(rows,key=lambda x:x[5]):
    tag='  <-- текущая' if i==0 else ''
    print('листья %2d lr %.2f дер %3d mc %2d cs %.1f l2 %4.1f   %11.0f %9.0f %11.0f %11.0f%s'%(
        p['num_leaves'],p['learning_rate'],p['n_estimators'],p['min_child_samples'],
        p['colsample_bytree'],p['reg_lambda'],r,sd,s,mix,tag))
