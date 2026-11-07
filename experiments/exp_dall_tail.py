import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings,pickle; pass; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from data import load_competition, canonical_smiles
from features import build
tr,te=load_competition(verbose=False); y=tr.y.to_numpy()
base=pickle.load(open(_CACHE + '/base.pkl','rb'))
e=pd.read_csv(_DATA + '/external.csv'); e['cs']=e.smiles.map(canonical_smiles)
e=e[e.cs.notna()&~e.cs.isin(set(tr.smiles))&~e.cs.isin(set(te.smiles))]
a=e[(e.SI>0)&(e.SI<10000)].copy(); ye=(a.SI>500).to_numpy().astype(int)
CL=dict(objective='binary',n_estimators=300,learning_rate=0.05,num_leaves=15,min_child_samples=10,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FPt={bl:build(tr.smiles.tolist(),bl)[0] for bl in REPR}
FPe={bl:build(a.cs.tolist(),bl)[0] for bl in REPR}
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
E_T=4286.
rows=[]
for tag in ('нет','desc','descall'):
    if tag=='нет': Dt=np.zeros((len(tr),0)); De=np.zeros((len(a),0))
    else:
        Dt,_=build(tr.smiles.tolist(),(tag,)); De,_=build(a.cs.tolist(),(tag,))
        k=(np.isfinite(Dt).all(0))&(np.isfinite(De).all(0))&(Dt.std(0)>0)&(De.std(0)>0)
        Dt,De=Dt[:,k],De[:,k]
    ps=[]
    for bl in REPR:
        Xe=np.hstack([FPe[bl],De]); Xt=np.hstack([FPt[bl],Dt]); kk=Xe.std(0)>0
        for s in range(3): ps.append(lgb.LGBMClassifier(random_state=s,**CL).fit(Xe[:,kk],ye).predict_proba(Xt[:,kk])[:,1])
    q=np.mean(ps,0); qq=q**2
    o={}
    for sch in ('random','scaffold'):
        o[sch]=np.mean([np.mean(((qq*E_T+(1-qq)*base[(sch,s)])[keep]-y[keep])**2) for s in range(4)])
    rows.append((tag,Dt.shape[1],roc_auc_score((y>500).astype(int),q),roc_auc_score((y>1000).astype(int),q),
                 o['random'],o['scaffold'],0.668*o['random']+0.332*o['scaffold']))
print('%-10s %6s %8s %8s %10s %10s %10s'%('дескр.','шт','AUC500','AUC1k','УСТ rnd','УСТ scf','УСТ смесь'))
for r in sorted(rows,key=lambda x:x[6]): print('%-10s %6d %8.4f %8.4f %10.0f %10.0f %10.0f'%r)
