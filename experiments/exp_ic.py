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
from binned_model import BinnedExpectationModel
tr,te=load_competition(verbose=False); y=tr.y.to_numpy(); S=tr.smiles.tolist()
def flat(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m)
tr['flat']=tr.smiles.map(flat)
r=pd.read_csv(_DATA + '/raw/chembl_activities.csv',low_memory=False)
U={'nM':1e-3,'uM':1.0,'M':1e6,'pM':1e-6,'mM':1e3}
r['uM']=pd.to_numeric(r.standard_value,errors='coerce')*r.standard_units.map(U)
r=r[r.uM.notna()&(r.uM>0)].copy()
vac=r.target_organism.astype(str).str.contains('vaccinia|Orthopox',case=False,na=False)|(r.is_vaccinia_assay==1)
pot=r[vac&r.standard_type.isin({'EC50','IC50'})].copy(); tox=r[r.standard_type.isin({'CC50','TC50','GI50','TD50'})].copy()
for d in (pot,tox): d['flat']=d.canonical_smiles.map(lambda s: flat(canonical_smiles(s)))
PIC=pot.groupby('flat')['uM'].median(); PCC=tox.groupby('flat')['uM'].median()
ic=tr.flat.map(PIC).to_numpy(dtype=float); cc=tr.flat.map(PCC).to_numpy(dtype=float)
EXTRA=np.c_[np.log10(np.where(np.isfinite(ic)&(ic>0),ic,np.nan)),
            np.log10(np.where(np.isfinite(cc)&(cc>0),cc,np.nan)),
            np.isfinite(ic).astype(float)]
print('строк с литературным IC50 %d | CC50 %d'%(np.isfinite(ic).sum(),np.isfinite(cc).sum()))
P=dict(objective='regression',n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FP={bl:build(S,bl)[0] for bl in REPR}
D,_=build(S,('descall',)); kd=(np.isfinite(D).all(0))&(D.std(0)>0); D=D[:,kd]
TT,_=build_title(tr.Title.tolist())
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
def run(use):
    o={}
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        R=[];F=[]
        for seed in range(3):
            b=np.zeros(len(tr))
            for trn,val in kf(S,n_splits=5,seed=seed):
                me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
                LG=[]
                for bl in REPR:
                    Xa=[FP[bl][trn],D[trn],A,TT[trn]]; Xb=[FP[bl][val],D[val],B,TT[val]]
                    if use: Xa.append(EXTRA[trn]); Xb.append(EXTRA[val])
                    m=lgb.LGBMRegressor(random_state=seed,**P).fit(np.hstack(Xa),np.log1p(y[trn]))
                    LG.append(np.clip(np.expm1(m.predict(np.hstack(Xb))),0,None))
                BN=np.mean([BinnedExpectationModel(blocks=tuple(list(bl)+['descall']),use_meta=True,use_title=True,seed=seed).fit(tr.iloc[trn]).predict(tr.iloc[val]) for bl in REPR],0)
                b[val]=0.5*np.mean(LG,0)+0.5*BN
            R.append(np.mean((b[keep]-y[keep])**2)); F.append(np.mean((b-y)**2))
        o[sname]=(np.mean(R),np.std(R),np.mean(F))
    return o
for nm,u in (('без литературного IC50',False),('+ литературные IC50/CC50',True)):
    o=run(u)
    print('%-26s УСТ rnd %9.0f ± %6.0f | УСТ scf %9.0f | УСТ смесь %9.0f | ПОЛН смесь %10.0f'%(
        nm,o['random'][0],o['random'][1],o['scaffold'][0],
        0.668*o['random'][0]+0.332*o['scaffold'][0],0.668*o['random'][2]+0.332*o['scaffold'][2]))
