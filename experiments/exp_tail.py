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
from sklearn.metrics import roc_auc_score
tr,te=load_competition(verbose=False); y=tr.y.to_numpy()
e=pd.read_csv(_DATA + '/external.csv'); e['smiles']=e.smiles.map(canonical_smiles)
e=e[e.smiles.notna()]; e=e[~e.smiles.isin(set(te.smiles))&~e.smiles.isin(set(tr.smiles))]
a=e[(e.SI>0)&(e.SI<10000)].copy()
THR=1000.
CL=dict(objective='binary',n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=10,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FPt={bl:build(tr.smiles.tolist(),bl)[0] for bl in REPR}
FPe={bl:build(a.smiles.tolist(),bl)[0] for bl in REPR}
ye=(a.SI>THR).to_numpy().astype(int); yt=(y>THR).astype(int)
print('положительных: внешних %d из %d | в train %d из %d'%(ye.sum(),len(ye),yt.sum(),len(yt)))
CORES={'камфан-амид':'CC1(C)C2CCC1(C)C(NC(=O))C2','камфан-амид2':'CC1(C)C2CCC(C)(C2)C1NC(=O)',
       'тековиримат':'O=C(NN1C(=O)C2C3CC(C4CC34)C2C1=O)','нуклеозид':'C1OC(CO)CC1'}
def fam(sm):
    out=[]
    pats=[(k,Chem.MolFromSmarts(v)) for k,v in CORES.items()]
    for s in sm:
        m=Chem.MolFromSmiles(s)
        out.append([1 if (m is not None and p is not None and m.HasSubstructMatch(p)) else 0 for _,p in pats])
    return np.array(out,float)
Ft=fam(tr.smiles.tolist()); Fe=fam(a.smiles.tolist())
print('семейства в train:',dict(zip(CORES,Ft.sum(0).astype(int))),'| доля высоких внутри:',
      [round(float(yt[Ft[:,i]>0].mean()),3) if (Ft[:,i]>0).sum() else None for i in range(Ft.shape[1])])
def qfun(mode):
    def f(trn,val,seed):
        ps=[]
        for bl in REPR:
            Xe,Xt=FPe[bl],FPt[bl]
            if mode=='ext':
                X,yy=Xe,ye
            elif mode=='both':
                X=np.vstack([Xe,Xt[trn]]); yy=np.r_[ye,yt[trn]]
            elif mode=='comp':
                X,yy=Xt[trn],yt[trn]
            if yy.sum()==0: ps.append(np.zeros(len(val))); continue
            k=X.std(0)>0
            ps.append(lgb.LGBMClassifier(random_state=seed,**CL).fit(X[:,k],yy).predict_proba(Xt[val][:,k])[:,1])
        return np.mean(ps,0)
    return f
for mode in ('ext','both','comp'):
    f=qfun(mode)
    for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
        aucs=[]
        for seed in range(4):
            p=np.zeros(len(tr))
            for trn,val in kf(tr['smiles'].tolist(),n_splits=5,seed=seed): p[val]=f(trn,val,seed)
            aucs.append(roc_auc_score(yt,p))
        print('  обучение %-5s | %-8s AUC(SI>1000) %.4f ± %.4f'%(mode,sname,np.mean(aucs),np.std(aucs)))
