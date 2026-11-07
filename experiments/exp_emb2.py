import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings; pass; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from rdkit import Chem, DataStructs, RDLogger; RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import rdFingerprintGenerator
from data import load_competition, canonical_smiles
from splits import random_kfold, scaffold_kfold
from features import build
from meta_features import MetaEncoder
from title_features import build_title
from binned_model import BinnedExpectationModel
from final_model_v2 import literature_tables, flat_smiles, _lit_features
tr,te=load_competition(verbose=False); y=tr.y.to_numpy(); S=tr.smiles.tolist()
tr['flat']=tr.smiles.map(flat_smiles)
T=literature_tables(_DATA + '/raw/chembl_activities.csv')
LIT=_lit_features(tr,T)
z=np.load(_DATA + '/embeddings.npz'); EMB=z['train']
print('эмбеддинги train: %s | литературных IC50: %d'%(EMB.shape,int(np.isfinite(LIT[:,0]).sum())))
cl=pd.read_csv(_DATA + '/external_clean.csv'); cl['cs']=cl.smiles.map(canonical_smiles)
cl=cl[cl.cs.notna()].copy(); cl['flat']=cl.cs.map(flat_smiles)
cl=cl[~cl.flat.isin(set(tr.flat.dropna()))]
g=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
def fp(s):
    m=Chem.MolFromSmiles(str(s)); return g.GetFingerprint(m) if m else None
hi=cl[cl.SI>500]; rf=[f for f in (fp(s) for s in hi.cs) if f is not None]
SIM=[]
for s in S:
    f=fp(s)
    if f is None: SIM.append([0,0,0]); continue
    sim=np.array(DataStructs.BulkTanimotoSimilarity(f,rf)); o=np.argsort(-sim)[:5]
    SIM.append([sim.max(),sim[o].mean(),float((sim>=.5).sum())])
SIM=np.array(SIM)
P=dict(objective='regression',n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FP={bl:build(S,bl)[0] for bl in REPR}
D,_=build(S,('descall',)); kd=(np.isfinite(D).all(0))&(D.std(0)>0); D=D[:,kd]
TT,_=build_title(tr.Title.tolist())
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
def run(extra):
    o={}
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        R=[]
        for seed in range(2):
            b=np.zeros(len(tr))
            for trn,val in kf(S,n_splits=5,seed=seed):
                me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
                LG=[]
                for bl in REPR:
                    Xa=[FP[bl][trn],D[trn],A,TT[trn],LIT[trn]]; Xb=[FP[bl][val],D[val],B,TT[val],LIT[val]]
                    for E_ in extra: Xa.append(E_[trn]); Xb.append(E_[val])
                    m=lgb.LGBMRegressor(random_state=seed,**P).fit(np.hstack(Xa),np.log1p(y[trn]))
                    LG.append(np.clip(np.expm1(m.predict(np.hstack(Xb))),0,None))
                BN=np.mean([BinnedExpectationModel(blocks=tuple(list(bl)+['descall']),use_meta=True,use_title=True,seed=seed).fit(tr.iloc[trn]).predict(tr.iloc[val]) for bl in REPR],0)
                b[val]=0.5*np.mean(LG,0)+0.5*BN
            R.append(np.mean((b[keep]-y[keep])**2))
        o[sname]=np.mean(R)
    return o
res=[]
for nm,e in (('текущая (desc+лит+близость)',[SIM]),('+ эмбеддинги',[SIM,EMB])):
    o=run(e); res.append((nm,o['random'],o['scaffold'],0.668*o['random']+0.332*o['scaffold']))
    print('  посчитано: %s'%nm)
print('\n%-28s %11s %11s %11s'%('вариант','УСТ rnd','УСТ scf','УСТ смесь'))
for r in sorted(res,key=lambda x:x[3]): print('%-28s %11.0f %11.0f %11.0f'%r)
