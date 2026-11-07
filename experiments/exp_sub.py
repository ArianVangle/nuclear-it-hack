import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
_DATA = str(_ROOT / "data")
_SUB = str(_ROOT / "submissions")
_CACHE = str(_ROOT / "experiments" / "_cache")
import sys,warnings,pickle; pass; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from data import load_competition, canonical_smiles
tr,te=load_competition(verbose=False); y=tr.y.to_numpy()
base=pickle.load(open(_CACHE + '/base.pkl','rb'))
def flat(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: return None
    Chem.RemoveStereochemistry(m); return Chem.MolToSmiles(m)
c=pd.read_csv('data/external_clean.csv'); c['cs']=c.smiles.map(canonical_smiles)
c=c[c.cs.notna()].copy(); c['flat']=c.cs.map(flat)
tr['flat']=tr.smiles.map(flat); te['flat']=te.smiles.map(flat)
exact=c.groupby('cs')['SI'].median(); fl=c.groupby('flat')['SI'].median()
for nm,d in (('train',tr),('test',te)):
    e1=d.smiles.isin(exact.index); e2=d.flat.isin(fl.index)
    print('%5s: точное совпадение %3d | по плоской структуре %3d | из %d (%.0f%%)'%(nm,e1.sum(),e2.sum(),len(d),100*e2.mean()))
ex=int(np.argmax(y)); keep=np.ones(len(tr),bool); keep[ex]=False
# значения-кандидаты для подстановки
v_exact=tr.smiles.map(exact).to_numpy(dtype=float)
v_flat=tr.flat.map(fl).to_numpy(dtype=float)
print('\nпокрытие train: точное %d | плоское %d'%(np.isfinite(v_exact).sum(),np.isfinite(v_flat).sum()))
# на train уже есть подстановка «та же молекула в train» — её не трогаем, это другое
def ev(p):
    return np.mean((p[keep]-y[keep])**2), np.mean((p-y)**2)
rows=[]
for sch in ('random','scaffold'):
    for nm,vv,cap in (('модель как есть',None,None),
                      ('подставить точное',v_exact,None),
                      ('подставить плоское',v_flat,None),
                      ('подставить плоское, потолок 5000',v_flat,5000.),
                      ('плоское, геом. среднее с моделью',v_flat,'geo')):
        R=[];F=[]
        for s in range(4):
            p=base[(sch,s)].copy()
            if vv is not None:
                m=np.isfinite(vv)
                if cap=='geo': p[m]=np.sqrt(np.maximum(vv[m],0)*np.maximum(p[m],1e-9))
                else:
                    w=np.clip(vv[m],0,cap) if cap else vv[m]
                    p[m]=w
            a,b=ev(p); R.append(a); F.append(b)
        rows.append((sch,nm,np.mean(R),np.mean(F)))
print('\n%-9s %-34s %12s %12s'%('схема','вариант','УСТ','ПОЛН'))
for sch in ('random','scaffold'):
    for r in sorted([x for x in rows if x[0]==sch],key=lambda z:z[2]):
        print('%-9s %-34s %12.0f %12.0f'%r)
print('\nсводно (0.668*random + 0.332*scaffold):')
d={}
for sch,nm,r,f in rows: d.setdefault(nm,{})[sch]=(r,f)
for nm,v in sorted(d.items(),key=lambda x:0.668*x[1]['random'][0]+0.332*x[1]['scaffold'][0]):
    print('  %-34s УСТ %10.0f | ПОЛН %11.0f'%(nm,0.668*v['random'][0]+0.332*v['scaffold'][0],
                                              0.668*v['random'][1]+0.332*v['scaffold'][1]))
