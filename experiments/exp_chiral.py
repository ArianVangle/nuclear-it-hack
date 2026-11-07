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
from binned_model import BinnedExpectationModel
tr,te=load_competition(verbose=False); y=tr.y.to_numpy(); S=tr.smiles.tolist()
raw=pd.read_csv(_DATA + '/train_v2.csv')
cc=pd.to_numeric(raw['CC50_uM'],errors='coerce').to_numpy()
ic=pd.to_numeric(raw['IC50_uM_VACV'],errors='coerce').to_numpy()
P=dict(objective='regression',n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
SETS={'текущий (без хиральности)':[('morgan',),('morganc',),('torsion',),('rdkfp',)],
      '+ хиральный морган':[('morgan',),('morganc',),('torsion',),('rdkfp',),('morganchi',)],
      'хиральный вместо обычного':[('morganchi',),('morganc',),('torsion',),('rdkfp',)]}
ALL=set(b for v in SETS.values() for b in v)
FP={bl:build(S,bl)[0] for bl in ALL}
D,_=build(S,('descall',)); kd=(np.isfinite(D).all(0))&(D.std(0)>0); D=D[:,kd]
TT,_=build_title(tr.Title.tolist())
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
def run(REPR):
    o={}
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        R=[]
        for seed in range(3):
            b=np.zeros(len(tr))
            for trn,val in kf(S,n_splits=5,seed=seed):
                me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
                LG=[]
                for bl in REPR:
                    Xa=np.hstack([FP[bl][trn],D[trn],A,TT[trn]]); Xb=np.hstack([FP[bl][val],D[val],B,TT[val]])
                    m=lgb.LGBMRegressor(random_state=seed,**P).fit(Xa,np.log1p(y[trn]))
                    LG.append(np.clip(np.expm1(m.predict(Xb)),0,None))
                BN=np.mean([BinnedExpectationModel(blocks=tuple(list(bl)+['descall']),use_meta=True,use_title=True,seed=seed).fit(tr.iloc[trn]).predict(tr.iloc[val]) for bl in REPR],0)
                main=0.5*np.mean(LG,0)+0.5*BN
                # разрешённая декомпозиция: цели только из train
                aux=[]
                for tgt in (cc,ic):
                    msk=np.isfinite(tgt)&(tgt>0); sel=np.intersect1d(trn,np.where(msk)[0])
                    ps=[]
                    for bl in REPR:
                        Xa=np.hstack([FP[bl][sel],D[sel],TT[sel]]); Xb=np.hstack([FP[bl][val],D[val],TT[val]])
                        ps.append(lgb.LGBMRegressor(random_state=seed,**P).fit(Xa,np.log10(tgt[sel])).predict(Xb))
                    aux.append(np.mean(ps,0))
                ratio=np.clip(10**(aux[0]-aux[1]),0,None)
                b[val]=0.5*main+0.5*ratio
            R.append(np.mean((b[keep]-y[keep])**2))
        o[sname]=(float(np.mean(R)),float(np.std(R)))
    return o
res=[]
for nm,R in SETS.items():
    o=run(R)
    mix=0.668*o['random'][0]+0.332*o['scaffold'][0]
    res.append((nm,o['random'][0],o['random'][1],o['scaffold'][0],mix))
    print('посчитано: %-28s смесь %9.0f'%(nm,mix),flush=True)
print('\n%-30s %11s %8s %11s %11s'%('набор представлений','УСТ rnd','±','УСТ scf','смесь'))
for r in sorted(res,key=lambda x:x[4]): print('%-30s %11.0f %8.0f %11.0f %11.0f'%r)
