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
tr['flat']=tr.smiles.map(flat); te['flat']=te.smiles.map(flat)
TF=set(tr.flat.dropna())|set(te.flat.dropna()); CS=set(tr.smiles)|set(te.smiles)
def prep(p):
    d=pd.read_csv(p); d['cs']=d.smiles.map(canonical_smiles); d=d[d.cs.notna()].copy()
    d['flat']=d.cs.map(flat)
    return d[~d.cs.isin(CS)&~d.flat.isin(TF)].copy()
OLD=prep(_DATA + '/external.csv'); OLD=OLD[(OLD.SI>0)&(OLD.SI<10000)]
NEW=prep(_DATA + '/external_clean.csv')
print('старый после строгого фильтра: %d | чистый: %d'%(len(OLD),len(NEW)))
CL=dict(objective='binary',n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=10,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
Dt,_=build(tr.smiles.tolist(),('desc',)); FPt={bl:build(tr.smiles.tolist(),bl)[0] for bl in REPR}
def mkq(src,thr):
    yy=(src.SI>thr).to_numpy().astype(int)
    if yy.sum()<12: return None,int(yy.sum())
    De,_=build(src.cs.tolist(),('desc',))
    kd=(np.isfinite(De).all(0))&(np.isfinite(Dt).all(0))&(De.std(0)>0)&(Dt.std(0)>0)
    ps=[]
    for bl in REPR:
        Xe=np.hstack([build(src.cs.tolist(),bl)[0],De[:,kd]]); Xt=np.hstack([FPt[bl],Dt[:,kd]]); k=Xe.std(0)>0
        for s in range(3): ps.append(lgb.LGBMClassifier(random_state=s,**CL).fit(Xe[:,k],yy).predict_proba(Xt[:,k])[:,1])
    return np.mean(ps,0),int(yy.sum())
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
rows=[]
for nm,src in (('старый',OLD),('чистый',NEW)):
    for thr in (100,500,1000):
        q,npos=mkq(src,thr)
        if q is None: print('  %s порог %d: всего %d положительных, пропуск'%(nm,thr,npos)); continue
        # величина хвоста по тому же набору
        ET=float(np.percentile(src.SI[src.SI>thr],75)); ET=min(ET,20000.)
        qq=q**2
        o={}
        for sch in ('random','scaffold'):
            o[sch]=np.mean([np.mean(((qq*ET+(1-qq)*base[(sch,s)])[keep]-y[keep])**2) for s in range(4)])
        rows.append(('%s, порог %4d'%(nm,thr),npos,ET,roc_auc_score((y>500).astype(int),q),
                     roc_auc_score((y>1000).astype(int),q),o['random'],o['scaffold'],
                     0.668*o['random']+0.332*o['scaffold']))
# базовая строка без хвоста
o={s:np.mean([np.mean((base[(s,i)][keep]-y[keep])**2) for i in range(4)]) for s in ('random','scaffold')}
rows.append(('без хвоста',0,0.,0.5,0.5,o['random'],o['scaffold'],0.668*o['random']+0.332*o['scaffold']))
print('\n%-20s %6s %8s %8s %8s %10s %10s %10s'%('набор','полож','E_TAIL','AUC500','AUC1k','УСТ rnd','УСТ scf','УСТ смесь'))
for r in sorted(rows,key=lambda x:x[7]): print('%-20s %6d %8.0f %8.4f %8.4f %10.0f %10.0f %10.0f'%r)
