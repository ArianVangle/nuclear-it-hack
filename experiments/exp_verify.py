import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings; pass; warnings.filterwarnings('ignore')
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
BASE=dict(objective='regression',verbose=-1,subsample=0.8,subsample_freq=1,
          n_estimators=400,learning_rate=0.05,num_leaves=7,colsample_bytree=0.8,reg_lambda=5.)
# СИДЫ 100..105 — другие, чем при подборе (0..1), чтобы проверка была независимой
SEEDS=list(range(100,106))
def run(mc):
    o={}
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        R=[]
        for seed in SEEDS:
            b=np.zeros(len(tr))
            for trn,val in kf(S,n_splits=5,seed=seed):
                me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
                ps=[]
                for bl in REPR:
                    Xa=np.hstack([FP[bl][trn],D[trn],A,TT[trn],LIT[trn]])
                    Xb=np.hstack([FP[bl][val],D[val],B,TT[val],LIT[val]])
                    m=lgb.LGBMRegressor(random_state=seed,min_child_samples=mc,**BASE).fit(Xa,np.log1p(y[trn]))
                    ps.append(np.clip(np.expm1(m.predict(Xb)),0,None))
                b[val]=np.mean(ps,0)
            R.append(np.mean((b[keep]-y[keep])**2))
        o[sname]=(float(np.mean(R)),float(np.std(R)),R)
    return o
res={}
for mc in (10,5):
    res[mc]=run(mc)
    print('посчитано min_child_samples=%d'%mc,flush=True)
print('\nнезависимая перепроверка на сидах 100..105:')
print('%-22s %11s %9s %11s %9s %11s'%('','УСТ rnd','±','УСТ scf','±','смесь'))
for mc in (10,5):
    o=res[mc]
    mix=0.668*o['random'][0]+0.332*o['scaffold'][0]
    print('min_child_samples=%-2d   %11.0f %9.0f %11.0f %9.0f %11.0f'%(
        mc,o['random'][0],o['random'][1],o['scaffold'][0],o['scaffold'][1],mix))
a=np.array(res[10]['random'][2]); b=np.array(res[5]['random'][2])
d=a-b
print('\nпопарная разница по сидам (random), положительная = mc=5 лучше:')
print('  ',[int(x) for x in d])
print('  среднее %.0f, стандартная ошибка %.0f, выиграл в %d случаях из %d'%(
    d.mean(),d.std(ddof=1)/np.sqrt(len(d)),int((d>0).sum()),len(d)))
a2=np.array(res[10]['scaffold'][2]); b2=np.array(res[5]['scaffold'][2]); d2=a2-b2
print('то же для scaffold: среднее %.0f, выиграл в %d из %d'%(d2.mean(),int((d2>0).sum()),len(d2)))
