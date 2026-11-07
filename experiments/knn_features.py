"""
Признаки по ближайшим соседям (Tanimoto на отпечатках Morgan).

Зачем. У 66,8% строк test каркас присутствует в train, то есть выборка
состоит из серий аналогов. Деревья на бинарных битах плохо выражают
«похоже на вот это конкретное соединение, у которого SI известен»,
а для серий аналогов это самый прямой сигнал.

Оракульный расчёт показал, где запас: при идеальном классификаторе заглушки
MSE упала бы с 13,0 до 4,69 млн, то есть почти весь остаток ошибки — это
неверное отнесение к классу. Поэтому соседские признаки считаются отдельно
для заглушек и для активных.

Утечки нет: на обучении сосед сам себе не сосед (leave-one-out),
на применении соседи берутся только из обучающей части фолда.
"""

import numpy as np
from rdkit import DataStructs, RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

RDLogger.DisableLog("rdApp.*")

_gen = GetMorganGenerator(radius=2, fpSize=2048)
TOPK = (1, 3, 5, 10)


def _fps(smiles_list):
    out = []
    for smi in smiles_list:
        mol = MolFromSmiles(smi) if isinstance(smi, str) else None
        out.append(_gen.GetFingerprint(mol) if mol is not None else None)
    return out


def _agg(sims, labels, logsi, is_sent, k):
    """Статистики по k ближайшим соседям."""
    if len(sims) == 0:
        return [0.0, 0.5, 0.0, 0.0]
    idx = np.argsort(sims)[::-1][:k]
    s = sims[idx]
    w = s + 1e-9
    sent = is_sent[idx]
    rate = float(np.average(sent, weights=w))
    act = idx[is_sent[idx] == 0]
    if len(act):
        wa = sims[act] + 1e-9
        mlog = float(np.average(logsi[act], weights=wa))
    else:
        mlog = 0.0
    return [float(s[0]), rate, mlog, float(s.mean())]


class KnnFeatures:
    """
    На каждое значение k даёт четыре числа: сходство с ближайшим соседом,
    взвешенная доля заглушек среди соседей, взвешенное среднее log10(SI)
    среди активных соседей, среднее сходство по k соседям.
    Плюс отдельно: сходство с ближайшей заглушкой и с ближайшим активным.
    """

    def __init__(self, topk=TOPK):
        self.topk = tuple(topk)

    def fit(self, df):
        self.fps_ = _fps(df["smiles"].tolist())
        self.is_sent_ = df["is_sentinel"].to_numpy(dtype=int)
        self.logsi_ = np.log10(np.clip(df["y"].to_numpy(dtype=float), 1e-3, None))
        self.valid_ = np.array([f is not None for f in self.fps_])
        self._ref = [f for f in self.fps_ if f is not None]
        self._ref_sent = self.is_sent_[self.valid_]
        self._ref_log = self.logsi_[self.valid_]
        return self

    def _rows(self, fps, drop_self_at=None):
        out = []
        for i, f in enumerate(fps):
            if f is None or not self._ref:
                out.append([0.0] * (4 * len(self.topk) + 2))
                continue
            sims = np.array(DataStructs.BulkTanimotoSimilarity(f, self._ref))
            if drop_self_at is not None:
                # leave-one-out: сама молекула не считается своим соседом
                j = drop_self_at[i]
                if j is not None:
                    sims[j] = -1.0
            row = []
            for k in self.topk:
                row += _agg(sims, None, self._ref_log, self._ref_sent, k)
            sent_mask = self._ref_sent == 1
            row.append(float(sims[sent_mask].max()) if sent_mask.any() else 0.0)
            row.append(float(sims[~sent_mask].max()) if (~sent_mask).any() else 0.0)
            out.append(row)
        return np.asarray(out, dtype=float)

    def fit_transform(self, df):
        self.fit(df)
        # позиция каждой обучающей молекулы среди опорных отпечатков
        pos, c = [], 0
        for ok in self.valid_:
            pos.append(c if ok else None)
            if ok:
                c += 1
        return self._rows(self.fps_, drop_self_at=pos)

    def transform(self, df):
        return self._rows(_fps(df["smiles"].tolist()))

    @property
    def names(self):
        out = []
        for k in self.topk:
            out += [f"knn{k}_sim_top1", f"knn{k}_sent_rate",
                    f"knn{k}_mean_logsi", f"knn{k}_sim_mean"]
        return out + ["knn_sim_nearest_sentinel", "knn_sim_nearest_active"]
