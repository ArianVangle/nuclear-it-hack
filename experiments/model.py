"""
Модель: классификатор заглушки + регрессор для активных + смешивание.

Центральное решение всего проекта — чем именно заполнять ответ.

MSE минимизируется условным математическим ожиданием, а не наиболее
вероятным значением. Если классификатор даёт p = 0.6, жёсткий выбор
«ставим 10000» при настоящем SI = 15 даёт ошибку порядка 10^8, тогда как
смесь p·10000 + (1-p)·m — примерно втрое меньше. На выборке, где каждый
пятый объект относится к классу заглушек, это и есть разница между первым
и десятым местом.

Отсюда следует второе: калибровка вероятностей важнее подбора
гиперпараметров. Некалиброванный бустинг систематически переуверен, и на
этой метрике переуверенность стоит дорого. Поэтому поверх внутреннего
scaffold-CV обучается изотоническая регрессия.

Регрессор учится на log10(SI), потому что распределение скошено на несколько
порядков. При обратном преобразовании среднее логарифма не равно логарифму
среднего, поэтому применяется поправка Дуана (smearing estimator).
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb
from features import build, drop_constant
from splits import scaffold_kfold

SENTINEL = 0.0

CLF_PARAMS = dict(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=7,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.3,
    reg_lambda=5.0,
    verbose=-1,
)

REG_PARAMS = dict(
    objective="regression",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=7,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.3,
    reg_lambda=5.0,
    verbose=-1,
)


class SentinelBlendModel:
    """
    fit(df) ожидает колонки: smiles, y, is_sentinel.
    predict(df) возвращает ожидаемое значение SI.
    """

    def __init__(self, blocks=("morgan", "desc"), clf_params=None, reg_params=None,
                 calibrate=True, inner_splits=5, log_clip_quantile=0.995,
                 smearing=True, use_meta=False, reg_target="log10",
                 reg_objective=None, use_knn=False, meta_cols=None,
                 use_flags=False, use_title=False, seed=0):
        self.blocks = blocks
        self.use_meta = use_meta
        self.clf_params = dict(CLF_PARAMS, **(clf_params or {}))
        self.reg_params = dict(REG_PARAMS, **(reg_params or {}))
        self.calibrate = calibrate
        self.inner_splits = inner_splits
        self.log_clip_quantile = log_clip_quantile
        self.smearing = smearing
        # "log10" — учим log10(SI), обратно с поправкой Дуана.
        # "raw"   — учим SI напрямую. Для MSE это теоретически правильнее:
        #           L2-регрессия оценивает условное СРЕДНЕЕ, а логарифм
        #           с обратным преобразованием даёт скорее медиану.
        self.reg_target = reg_target
        self.reg_objective = reg_objective
        self.use_knn = use_knn
        self.meta_cols = meta_cols
        self.use_flags = use_flags
        self.use_title = use_title
        self.seed = seed

    # ------------------------------------------------------------- обучение

    def fit(self, df, sample_weight=None):
        smiles = df["smiles"].tolist()
        X, names = build(smiles, self.blocks)
        self._names = names
        self._keep = X.std(axis=0) > 0
        X = X[:, self._keep]

        # Признаки из strain и Culture. На обучении кодировка считается
        # вне фолда, иначе модель увидит собственный ответ через признак.
        self.meta_ = None
        if self.use_meta:
            from meta_features import MetaEncoder
            from meta_features import META_COLS
            self.meta_ = MetaEncoder(cols=self.meta_cols or META_COLS,
                                     inner_splits=self.inner_splits,
                                     use_flags=self.use_flags, seed=self.seed)
            X = np.hstack([X, self.meta_.fit_transform(df)])

        if self.use_title:
            from title_features import build_title
            T, _ = build_title(df["Title"].tolist())
            X = np.hstack([X, T])

        # Признаки по ближайшим соседям. На обучении — leave-one-out.
        self.knn_ = None
        if self.use_knn:
            from knn_features import KnnFeatures
            self.knn_ = KnnFeatures()
            X = np.hstack([X, self.knn_.fit_transform(df)])

        y = df["y"].to_numpy(dtype=float)
        s = df["is_sentinel"].to_numpy(dtype=int)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        self._prior = float(s.mean())
        self._mean_active = float(y[s == 0].mean()) if (s == 0).any() else SENTINEL

        # --- классификатор заглушки
        self.clf_ = lgb.LGBMClassifier(random_state=self.seed, **self.clf_params)
        self.clf_.fit(X, s, sample_weight=w)

        # --- калибровка на внутреннем scaffold-CV
        self.calibrator_ = None
        if self.calibrate and 0 < s.sum() < len(s):
            oof = np.full(len(s), np.nan)
            for tr, va in scaffold_kfold(smiles, self.inner_splits, seed=self.seed + 100):
                if len(np.unique(s[tr])) < 2:
                    oof[va] = s[tr].mean()
                    continue
                m = lgb.LGBMClassifier(random_state=self.seed, **self.clf_params)
                m.fit(X[tr], s[tr], sample_weight=None if w is None else w[tr])
                oof[va] = m.predict_proba(X[va])[:, 1]
            if np.isfinite(oof).all():
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(oof, s)
                self.calibrator_ = iso

        # --- регрессор на активных, в log10
        act = s == 0
        self.reg_ = None
        self.smear_ = 1.0
        if act.sum() >= 20:
            params = dict(self.reg_params)
            if self.reg_objective:
                params["objective"] = self.reg_objective

            if self.reg_target == "raw":
                ya = np.clip(y[act], 0.0, None)
                hi = np.quantile(ya, self.log_clip_quantile)
                ya_clipped = np.clip(ya, 0.0, hi)
                self.reg_ = lgb.LGBMRegressor(random_state=self.seed, **params)
                self.reg_.fit(X[act], ya_clipped,
                              sample_weight=None if w is None else w[act])
                return self

            ya = np.log10(np.clip(y[act], 1e-3, None))
            hi = np.quantile(ya, self.log_clip_quantile)
            lo = np.quantile(ya, 1 - self.log_clip_quantile)
            ya_clipped = np.clip(ya, lo, hi)

            self.reg_ = lgb.LGBMRegressor(random_state=self.seed, **params)
            self.reg_.fit(X[act], ya_clipped, sample_weight=None if w is None else w[act])

            if self.smearing:
                Xa, sa, ya_full = X[act], [smiles[i] for i in np.where(act)[0]], ya
                oof_log = np.full(act.sum(), np.nan)
                for tr, va in scaffold_kfold(sa, self.inner_splits, seed=self.seed + 200):
                    m = lgb.LGBMRegressor(random_state=self.seed, **params)
                    m.fit(Xa[tr], np.clip(ya_full[tr], lo, hi))
                    oof_log[va] = m.predict(Xa[va])
                resid = ya_full - oof_log
                resid = resid[np.isfinite(resid)]
                if len(resid):
                    self.smear_ = float(np.clip(np.mean(10.0 ** resid), 0.5, 20.0))
        return self

    # ---------------------------------------------------------- предсказание

    def predict_parts(self, df):
        """Возвращает (p_заглушка, ожидаемый_SI_если_активна)."""
        smiles = df["smiles"].tolist()
        X, _ = build(smiles, self.blocks)
        X = X[:, self._keep]
        if self.meta_ is not None:
            X = np.hstack([X, self.meta_.transform(df)])
        if self.use_title:
            from title_features import build_title
            T, _ = build_title(df["Title"].tolist())
            X = np.hstack([X, T])
        if self.knn_ is not None:
            X = np.hstack([X, self.knn_.transform(df)])

        p = self.clf_.predict_proba(X)[:, 1]
        if self.calibrator_ is not None:
            p = self.calibrator_.predict(p)
        p = np.clip(p, 0.0, 1.0)

        if self.reg_ is not None:
            raw = self.reg_.predict(X)
            m = raw if self.reg_target == "raw" else (10.0 ** raw) * self.smear_
        else:
            m = np.full(len(df), self._mean_active)
        m = np.clip(m, 0.0, None)
        return p, m

    def predict(self, df):
        """Условное среднее — то, что минимизирует MSE."""
        p, m = self.predict_parts(df)
        return p * SENTINEL + (1.0 - p) * m


# ------------------------------------------------------------- для стенда


def make_fit_predict(**kwargs):
    """Оборачивает модель в интерфейс, который ждёт evaluate.repeated_cv."""
    def fit_predict(train_df, val_df, seed):
        model = SentinelBlendModel(seed=seed, **kwargs)
        model.fit(train_df)
        return model.predict(val_df)
    return fit_predict


def make_fit_predict_hard(**kwargs):
    """Вариант с жёстким выбором вместо смеси — для доказательства, что смесь лучше."""
    def fit_predict(train_df, val_df, seed):
        model = SentinelBlendModel(seed=seed, **kwargs)
        model.fit(train_df)
        p, m = model.predict_parts(val_df)
        return np.where(p > 0.5, SENTINEL, m)
    return fit_predict
