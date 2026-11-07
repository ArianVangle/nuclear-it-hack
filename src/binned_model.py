"""
Предсказание условного среднего через распределение по интервалам.

Зачем это вместо связки «классификатор заглушки + регрессор».

MSE минимизируется условным средним E[SI | x]. Текущая модель собирает его
из двух частей: вероятности заглушки и точечного прогноза регрессора,
обученного в логарифме и преобразованного обратно. Это приближение,
и у него две слабости. Во-первых, регрессор в логарифме оценивает скорее
условную медиану, а поправка Дуана чинит смещение только в среднем по
выборке, а не для каждой молекулы. Во-вторых, тяжёлый хвост приходится
обрезать потолком, из-за чего молекулы с действительно высоким SI
систематически недооцениваются — а именно они дороже всего стоят при MSE.

Здесь вместо этого оценивается всё распределение. Шкала SI разбивается
на интервалы, многоклассовый классификатор даёт вероятность каждого,
и среднее считается напрямую:

    E[SI | x] = сумма по интервалам k от P(k | x) * E[SI | интервал k]

Заглушка становится обычным классом, а хвост — отдельным классом со своим
большим представителем. Обрезать ничего не нужно: вклад хвоста в среднее
возникает сам, пропорционально его вероятности.
"""

import numpy as np
import lightgbm as lgb

from features import build

SENTINEL = 0.0

# Границы интервалов по SI. Подобраны по распределению обучающей выборки:
# медиана активных около 6, три четверти ниже 100, выше 1000 — единицы молекул.
# Заглушка 10000 вынесена в отдельный класс, и ещё один класс — всё выше неё.
# Нулевой SI — отдельный класс (индекс 0), далее интервалы по возрастанию.
# Верхний класс — хвост выше 1000, где сосредоточена почти вся дисперсия.
EDGES = [1e-9, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]

CLF_PARAMS = dict(
    objective="multiclass",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=7,
    min_child_samples=5,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    verbose=-1,
)


def assign_bins(y):
    """Класс 0 — нулевой SI, далее интервалы, последний — хвост."""
    y = np.asarray(y, dtype=float)
    k = np.digitize(y, EDGES, right=False)   # 0 для y<1e-9, далее по границам
    return k.astype(int)


N_BINS = len(EDGES) + 1


class BinnedExpectationModel:
    """
    fit(df) ожидает колонки smiles, y, is_sentinel (+ strain/Culture при use_meta).
    predict(df) возвращает E[SI | x].
    """

    def __init__(self, blocks=("morgan",), params=None, use_meta=True,
                 meta_cols=None, inner_splits=5, use_title=False, seed=0):
        self.use_title = use_title
        self.blocks = blocks
        self.params = dict(CLF_PARAMS, **(params or {}))
        self.use_meta = use_meta
        self.meta_cols = meta_cols
        self.inner_splits = inner_splits
        self.seed = seed

    def _matrix(self, df, fit=False):
        X, _ = build(df["smiles"].tolist(), self.blocks)
        if fit:
            self._keep = X.std(axis=0) > 0
        X = X[:, self._keep]
        if self.use_meta:
            if fit:
                from meta_features import MetaEncoder, META_COLS
                self.meta_ = MetaEncoder(cols=self.meta_cols or META_COLS,
                                         inner_splits=self.inner_splits,
                                         seed=self.seed)
                X = np.hstack([X, self.meta_.fit_transform(df)])
            else:
                X = np.hstack([X, self.meta_.transform(df)])
        if self.use_title:
            from title_features import build_title
            T, _ = build_title(df["Title"].tolist())
            X = np.hstack([X, T])
        return X

    def fit(self, df, sample_weight=None):
        X = self._matrix(df, fit=True)
        y = df["y"].to_numpy(dtype=float)
        k = assign_bins(y)

        # Представитель интервала — среднее попавших в него значений.
        # Для пустых интервалов берём середину, иначе класс, не встретившийся
        # в обучении, не смог бы дать вклад в среднее.
        reps = np.zeros(N_BINS)
        for b in range(N_BINS):
            m = k == b
            if m.any():
                reps[b] = y[m].mean()
            elif b == 0:
                reps[b] = 0.0
            elif b < len(EDGES):
                reps[b] = 0.5 * (EDGES[b - 1] + EDGES[b])
            else:
                reps[b] = EDGES[-1] * 2
        self.reps_ = reps

        # LightGBM требует подряд идущие метки классов
        self.classes_ = np.unique(k)
        remap = {c: i for i, c in enumerate(self.classes_)}
        k_dense = np.array([remap[v] for v in k])

        params = dict(self.params)
        params["num_class"] = len(self.classes_)
        self.clf_ = lgb.LGBMClassifier(random_state=self.seed, **params)
        self.clf_.fit(X, k_dense, sample_weight=sample_weight)
        return self

    def predict_proba_bins(self, df):
        X = self._matrix(df)
        P = self.clf_.predict_proba(X)
        full = np.zeros((len(df), N_BINS))
        for i, c in enumerate(self.classes_):
            full[:, c] = P[:, i]
        return full

    def predict(self, df):
        """Условное среднее — ровно то, что минимизирует MSE."""
        P = self.predict_proba_bins(df)
        return P @ self.reps_


def make_fit_predict(**kwargs):
    def fit_predict(train_df, val_df, seed):
        m = BinnedExpectationModel(seed=seed, **kwargs)
        m.fit(train_df)
        return m.predict(val_df)
    return fit_predict
