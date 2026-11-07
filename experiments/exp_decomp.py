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
from final_model_v2 import literature_tables, rebuilt_tables, merge_tables, flat_smiles, _lit_features
tr,te=load_competition(verbose=False); y=tr.y.to_numpy(); S=tr.smiles.tolist()
tr['flat']=tr.smiles.map(flat_smiles)
# ВАЖНО: CC50/IC50 берутся ТОЛЬКО из train и только как ЦЕЛИ вспомогательных
# моделей. Из test они не читаются нигде — это прямо запрещено правилами.
raw=pd.read_csv(_DATA + '/train_v2.csv')
cc=pd.to_numeric(raw['CC50_uM'],errors='coerce').to_numpy()
ic=pd.to_numeric(raw['IC50_uM_VACV'],errors='coerce').to_numpy()
okc=np.isfinite(cc)&(cc>0); oki=np.isfinite(ic)&(ic>0)
print('в train: CC50 у %d строк | IC50 у %d | обе у %d'%(okc.sum(),oki.sum(),(okc&oki).sum()))
T=merge_tables(literature_tables(_DATA + '/raw/chembl_activities.csv'),
               rebuilt_tables(_DATA + '/external_rebuilt.csv'))
LIT=_lit_features(tr,T)
P=dict(objective='regression',n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=10,
       subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5.,verbose=-1)
REPR=[('morgan',),('morganc',),('torsion',),('rdkfp',)]
FP={bl:build(S,bl)[0] for bl in REPR}
D,_=build(S,('descall',)); kd=(np.isfinite(D).all(0))&(D.std(0)>0); D=D[:,kd]
TT,_=build_title(tr.Title.tolist())
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
def fold_predict(trn,val,seed):
    me=MetaEncoder(seed=seed); A=me.fit_transform(tr.iloc[trn]); B=me.transform(tr.iloc[val])
    def X(idx,M): return np.hstack([FP[M][idx],D[idx],(A if idx is trn else B),TT[idx],LIT[idx]])
    # основная модель
    LG=[]
    for bl in REPR:
        Xa=np.hstack([FP[bl][trn],D[trn],A,TT[trn],LIT[trn]]); Xb=np.hstack([FP[bl][val],D[val],B,TT[val],LIT[val]])
        m=lgb.LGBMRegressor(random_state=seed,**P).fit(Xa,np.log1p(y[trn]))
        LG.append(np.clip(np.expm1(m.predict(Xb)),0,None))
    BN=np.mean([BinnedExpectationModel(blocks=tuple(list(bl)+['descall']),use_meta=True,use_title=True,seed=seed).fit(tr.iloc[trn]).predict(tr.iloc[val]) for bl in REPR],0)
    main=0.5*np.mean(LG,0)+0.5*BN
    # вспомогательные модели на CC50 и IC50 (цели только из train)
    def aux(mask,target):
        sel=np.intersect1d(trn,np.where(mask)[0])
        if len(sel)<30: return None
        ps=[]
        for bl in REPR:
            Xa=np.hstack([FP[bl][sel],D[sel],TT[sel],LIT[sel]]); Xb=np.hstack([FP[bl][val],D[val],TT[val],LIT[val]])
            m=lgb.LGBMRegressor(random_state=seed,**P).fit(Xa,np.log10(target[sel]))
            ps.append(m.predict(Xb))
        return np.mean(ps,0)
    lc=aux(okc,cc); li=aux(oki,ic)
    if lc is None or li is None: return main,None
    return main, np.clip(10**(lc-li),0,None)
res={}
for sname,kf in (('random',random_kfold),('scaffold',scaffold_kfold)):
    acc={'основная':[],'декомпозиция':[],'смесь 50/50':[],'смесь 75/25':[]}
    for seed in range(3):
        M=np.zeros(len(tr)); Dc=np.zeros(len(tr))
        for trn,val in kf(S,n_splits=5,seed=seed):
            m,d=fold_predict(trn,val,seed); M[val]=m; Dc[val]=d if d is not None else m
        for nm,p in (('основная',M),('декомпозиция',Dc),('смесь 50/50',0.5*M+0.5*Dc),('смесь 75/25',0.75*M+0.25*Dc)):
            acc[nm].append(np.mean((p[keep]-y[keep])**2))
    res[sname]={k:float(np.mean(v)) for k,v in acc.items()}
    print('посчитано:',sname)
print('\n%-16s %11s %11s %11s'%('вариант','УСТ rnd','УСТ scf','УСТ смесь'))
rows=[(k,res['random'][k],res['scaffold'][k],0.668*res['random'][k]+0.332*res['scaffold'][k]) for k in res['random']]
for r in sorted(rows,key=lambda x:x[3]): print('%-16s %11.0f %11.0f %11.0f'%r)
