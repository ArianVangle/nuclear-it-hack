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
tr,te=load_competition(verbose=False); y=tr.y.to_numpy(); S=tr.smiles.tolist()
def flat(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m)
tr['flat']=tr.smiles.map(flat)
# --- признаки близости к чистым противооспенным веществам ---
cl=pd.read_csv(_DATA + '/external_clean.csv'); cl['cs']=cl.smiles.map(canonical_smiles)
cl=cl[cl.cs.notna()].copy(); cl['flat']=cl.cs.map(flat)
cl=cl[~cl.flat.isin(set(tr.flat.dropna()))]        # без двойников, честно
g=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
def fp(s):
    m=Chem.MolFromSmiles(str(s)); return g.GetFingerprint(m) if m else None
def simfeat(smiles, ref, w):
    rf=[fp(s) for s in ref]; ok=[i for i,f in enumerate(rf) if f is not None]
    rf=[rf[i] for i in ok]; w=np.asarray(w)[ok]
    out=[]
    for s in smiles:
        f=fp(s)
        if f is None: out.append([0,0,0,0]); continue
        sim=np.array(DataStructs.BulkTanimotoSimilarity(f,rf))
        o=np.argsort(-sim)[:5]
        out.append([sim.max(), sim[o].mean(), float((sim>=.5).sum()), float((sim*np.log1p(w)).max())])
    return np.array(out)
hi=cl[cl.SI>500]
print('эталонных высокоселективных (без двойников): %d'%len(hi))
SIM=simfeat(S,hi.cs.tolist(),hi.SI.tolist())
print('признак «макс Tanimoto»: медиана %.2f, доля >=0.7 %.1f%%'%(np.median(SIM[:,0]),100*(SIM[:,0]>=.7).mean()))
# --- эмбеддинги ---
try:
    z=np.load(_DATA + '/embeddings.npz'); EMB=z[z.files[0]]
    print('эмбеддинги: %s'%(EMB.shape,))
    if EMB.shape[0]!=len(tr)+len(te): EMB=None; print('  размер не совпал, пропускаю')
    else: EMB=EMB[:len(tr)]
except Exception as e:
    EMB=None; print('эмбеддинги недоступны:',e)
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
        for seed in range(3):
            b=np.zeros(len(tr))
            for trn,val in kf(S,n_splits=5,seed=seed):
                me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
                LG=[]
                for bl in REPR:
                    Xa=[FP[bl][trn],D[trn],A,TT[trn]]; Xb=[FP[bl][val],D[val],B,TT[val]]
                    for E_ in extra: Xa.append(E_[trn]); Xb.append(E_[val])
                    m=lgb.LGBMRegressor(random_state=seed,**P).fit(np.hstack(Xa),np.log1p(y[trn]))
                    LG.append(np.clip(np.expm1(m.predict(np.hstack(Xb))),0,None))
                BN=np.mean([BinnedExpectationModel(blocks=tuple(list(bl)+['descall']),use_meta=True,use_title=True,seed=seed).fit(tr.iloc[trn]).predict(tr.iloc[val]) for bl in REPR],0)
                b[val]=0.5*np.mean(LG,0)+0.5*BN
            R.append(np.mean((b[keep]-y[keep])**2))
        o[sname]=np.mean(R)
    return o
V=[('базовая',[]),('+ близость к противооспенным',[SIM])]
if EMB is not None: V.append(('+ эмбеддинги',[EMB]))
V.append(('+ близость и эмбеддинги',[SIM]+([EMB] if EMB is not None else [])))
res=[]
for nm,e in V:
    o=run(e); res.append((nm,o['random'],o['scaffold'],0.668*o['random']+0.332*o['scaffold']))
print('\n%-32s %11s %11s %11s'%('вариант','УСТ rnd','УСТ scf','УСТ смесь'))
for r in sorted(res,key=lambda x:x[3]): print('%-32s %11.0f %11.0f %11.0f'%r)
