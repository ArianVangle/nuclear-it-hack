import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings; pass; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from data import load_competition, canonical_smiles
from splits import random_kfold, scaffold_kfold
from features import build
from meta_features import MetaEncoder
from title_features import build_title
tr,te=load_competition(verbose=False); y=tr.y.to_numpy(); S=tr.smiles.tolist()
def flat(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m)
tr['flat']=tr.smiles.map(flat); te['flat']=te.smiles.map(flat)
c=pd.read_csv(_DATA + '/external_clean.csv'); c['cs']=c.smiles.map(canonical_smiles)
c=c[c.cs.notna()].copy(); c['flat']=c.cs.map(flat)
EXT=c[~c.flat.isin(set(te.flat.dropna()))].reset_index(drop=True)
print('внешних обучающих молекул: %d (SI>1000 у %d)'%(len(EXT),(EXT.SI>1000).sum()))
P=dict(objective='regression',n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morganchi',),('morgancchi',),('torsionchi',),('rdkfp',)]
FPt={bl:build(S,bl)[0] for bl in REPR}
FPe={bl:build(EXT.cs.tolist(),bl)[0] for bl in REPR}
Dt,_=build(S,('descall',)); De,_=build(EXT.cs.tolist(),('descall',))
kd=(np.isfinite(Dt).all(0))&(np.isfinite(De).all(0))&(Dt.std(0)>0); Dt,De=Dt[:,kd],De[:,kd]
TTt,_=build_title(tr.Title.tolist())
ye=np.log1p(EXT.SI.to_numpy())
ext_flat=EXT.flat.to_numpy()
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
def run(use_ext, weight=1.0):
    o={}
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        R=[]
        for seed in range(3):
            b=np.zeros(len(tr))
            for trn,val in kf(S,n_splits=5,seed=seed):
                me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
                # внешние молекулы, совпадающие с ВАЛИДАЦИЕЙ, убираем — иначе утечка в CV
                vflat=set(tr.flat.iloc[val].dropna())
                em=~pd.Series(ext_flat).isin(vflat).to_numpy()
                ps=[]
                for bl in REPR:
                    Xa=np.hstack([FPt[bl][trn],Dt[trn],A,TTt[trn]])
                    Xb=np.hstack([FPt[bl][val],Dt[val],B,TTt[val]])
                    ya=np.log1p(y[trn]); w=np.ones(len(trn))
                    if use_ext:
                        nanA=np.full((em.sum(),A.shape[1]),np.nan)
                        nanT=np.full((em.sum(),TTt.shape[1]),np.nan)
                        Xa=np.vstack([Xa,np.hstack([FPe[bl][em],De[em],nanA,nanT])])
                        ya=np.r_[ya,ye[em]]; w=np.r_[w,np.full(em.sum(),weight)]
                    m=lgb.LGBMRegressor(random_state=seed,**P).fit(Xa,ya,sample_weight=w)
                    ps.append(np.clip(np.expm1(m.predict(Xb)),0,None))
                b[val]=np.mean(ps,0)
            R.append(np.mean((b[keep]-y[keep])**2))
        o[sname]=(float(np.mean(R)),float(np.std(R)))
    return o
res=[]
for nm,ue,w in (('только соревнование',False,1.),('+ внешние, вес 1.0',True,1.),
                ('+ внешние, вес 0.5',True,.5),('+ внешние, вес 0.25',True,.25)):
    o=run(ue,w); mix=0.668*o['random'][0]+0.332*o['scaffold'][0]
    res.append((nm,o['random'][0],o['random'][1],o['scaffold'][0],mix))
    print('посчитано: %-24s смесь %9.0f'%(nm,mix),flush=True)
print('\n%-24s %11s %8s %11s %11s'%('вариант','УСТ rnd','±','УСТ scf','смесь'))
for r in sorted(res,key=lambda x:x[4]): print('%-24s %11.0f %8.0f %11.0f %11.0f'%r)
