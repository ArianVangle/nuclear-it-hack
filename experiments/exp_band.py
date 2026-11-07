import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings,pickle; pass; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from sklearn.metrics import roc_auc_score
from data import load_competition, canonical_smiles
from features import build
tr,te=load_competition(verbose=False); y=tr.y.to_numpy()
base=pickle.load(open(_CACHE + '/base.pkl','rb'))
def flat(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m)
e=pd.read_csv(_DATA + '/external.csv'); e['cs']=e.smiles.map(canonical_smiles)
e=e[e.cs.notna()].copy(); e['flat']=e.cs.map(flat)
tr['flat']=tr.smiles.map(flat); te['flat']=te.smiles.map(flat)
TF=set(tr.flat.dropna())|set(te.flat.dropna())
ex_strict=e[~e.cs.isin(set(tr.smiles))&~e.cs.isin(set(te.smiles))&~e.flat.isin(TF)]
ex_loose =e[~e.cs.isin(set(tr.smiles))&~e.cs.isin(set(te.smiles))]
CL=dict(objective='binary',n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=10,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
Dt,_=build(tr.smiles.tolist(),('desc',)); FPt={bl:build(tr.smiles.tolist(),bl)[0] for bl in REPR}
def mkq(src,lab,name):
    a=src[(src.SI>0)&(src.SI<10000)].copy()
    yy=lab(a.SI.to_numpy()).astype(int)
    if yy.sum()<10: return None,0
    De,_=build(a.cs.tolist(),('desc',))
    kd=(np.isfinite(De).all(0))&(np.isfinite(Dt).all(0))&(De.std(0)>0)&(Dt.std(0)>0)
    ps=[]
    for bl in REPR:
        Xe=np.hstack([build(a.cs.tolist(),bl)[0],De[:,kd]]); Xt=np.hstack([FPt[bl],Dt[:,kd]]); k=Xe.std(0)>0
        for s in range(3): ps.append(lgb.LGBMClassifier(random_state=s,**CL).fit(Xe[:,k],yy).predict_proba(Xt[:,k])[:,1])
    return np.mean(ps,0), int(yy.sum())
LABS={'>500 (текущая)':lambda s:s>500,
      '>100':lambda s:s>100,
      'полоса 100..2000':lambda s:(s>100)&(s<2000),
      'полоса 50..2000':lambda s:(s>50)&(s<2000),
      'полоса 100..5000':lambda s:(s>100)&(s<5000)}
ex_=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex_]=False
E_T=2220.
rows=[]
for sname_src,src in (('строгий',ex_strict),('текущий',ex_loose)):
    for ln,lf in LABS.items():
        q,npos=mkq(src,lf,ln)
        if q is None: continue
        a500=roc_auc_score((y>500).astype(int),q); a1k=roc_auc_score((y>1000).astype(int),q)
        qq=q**2
        o={}
        for sch in ('random','scaffold'):
            R=[np.mean(((qq*E_T+(1-qq)*base[(sch,s)])[keep]-y[keep])**2) for s in range(4)]
            o[sch]=np.mean(R)
        rows.append(('%s | %s'%(sname_src,ln),npos,a500,a1k,o['random'],o['scaffold'],0.668*o['random']+0.332*o['scaffold']))
print('%-34s %5s %7s %7s %10s %10s %10s'%('набор | разметка','полож','AUC500','AUC1k','УСТ rnd','УСТ scf','УСТ смесь'))
for r in sorted(rows,key=lambda x:x[6]): print('%-34s %5d %7.4f %7.4f %10.0f %10.0f %10.0f'%r)
