"""
Признаки из полей strain и Culture.

Организаторы подтвердили в чате, что использовать их можно.

Почему они работают. В train доля заглушек по группам Culture доходит до
крайностей: «HELA cells» — 100% заглушек (11 из 11), «MT-4» — 100%,
«lung fibroblast cells» — 85%, тогда как «HeLa cells», «PRK cells»,
«A549 cells» — ровно 0%. Разный регистр у HeLa и HELA при противоположном
поведении показывает, что это не биологическое свойство клеточной линии,
а фактически метка публикации: разные статьи писали название по-разному.
Поэтому строки НЕ нормализуются — различие в написании само по себе несёт
информацию.

Все 18 значений Culture из test присутствуют в train, у strain невиданных
значений 1,5% строк. Поэтому кодирование по обучающей выборке переносится
на test почти без потерь.

Кодирование — сглаженное target encoding, посчитанное вне фолда (out-of-fold),
иначе модель запомнит собственную разметку и оценка окажется завышенной.
"""

import numpy as np
import pandas as pd

from splits import scaffold_kfold

# ОТКАЧЕНО К strain + Culture по результату лидерборда.
#
# Совместный ключ strain||Culture локально выглядел лучше (random 11,23
# против 11,83 млн; scaffold 14,92 против 15,44), но на лидерборде дал
# СТАБИЛЬНОЕ УХУДШЕНИЕ: 10 192 210 против 10 098 547 в одиночку
# и 10 211 981 вместе с потолком.
#
# Причина в том, что target encoding по мелкому ключу запоминает состав
# групп. В кросс-валидации проверочные молекулы берутся из того же пула,
# что и источник кодировки, поэтому групповая статистика переносится
# идеально. На тесте группы те же по названию, но состав внутри другой:
# пара, где в train заглушек 100%, в тесте может давать 60%. Чем мельче
# ключ, тем сильнее эффект; отдельные strain и Culture крупнее и устойчивее.
#
# Вывод шире одного признака: для target encoding локальная оценка
# систематически оптимистична, сколько повторов ни делай.
META_COLS = ["strain", "Culture"]

# Бинарные метки, найденные прямым разбором текста. Доля заглушек в train:
# Western Reserve, Lister, IHD, BSC, ATCC, variola — 0%; fibroblast — 79%;
# ectromelia — 40%, HeLa — 39%. Это части названий штаммов, клеточных линий
# и упоминания родственных вирусов, то есть реальная структура внутри строки.
TEXT_FLAGS = {
    "copenhagen": r"copenhagen",
    "western_reserve": r"western\s+reserve|\bWR\b",
    "lister": r"lister",
    "ihd": r"\bIHD\b",
    "cowpox": r"cowpox",
    "ectromelia": r"ectromelia|ECTV",
    "variola": r"variola",
    "vero": r"vero",
    "hela": r"hela",
    "bsc": r"bsc",
    "hel": r"\bhel\b",
    "fibroblast": r"fibroblast",
    "atcc": r"atcc",
    "mt4": r"\bMT-?4\b",
    "a549": r"a549",
    "cem": r"\bcem\b",
}
# Чем меньше группа, тем сильнее её оценка тянется к общей доле.
# Значение 3 подобрано по OOF: при 10 сигнал от групп в 8–11 молекул
# (HELA cells, MT-4) размывается почти вдвое, при 1 — переобучение.
SMOOTHING = 3.0


def _smoothed_rate(counts, sums, prior, k):
    return (sums + prior * k) / (counts + k)


class MetaEncoder:
    """
    Кодирует strain и Culture тремя числами на колонку:
      - сглаженная доля заглушек в группе,
      - сглаженное среднее log10(SI) среди активных,
      - размер группы.
    """

    def __init__(self, cols=META_COLS, smoothing=None, inner_splits=5,
                 use_flags=False, seed=0):
        self.use_flags = use_flags
        self.cols = list(cols)
        # значение по умолчанию читается в момент вызова, а не при
        # определении класса, иначе его нельзя поменять извне
        self.smoothing = SMOOTHING if smoothing is None else smoothing
        self.inner_splits = inner_splits
        self.seed = seed

    # ------------------------------------------------------------- обучение

    @staticmethod
    def _prepare(df):
        """Совместный ключ двух полей: он почти детерминирует заглушку."""
        d = df.copy()
        d["_joint"] = d.get("strain", "").astype(str) + "||" + d.get("Culture", "").astype(str)
        return d

    @staticmethod
    def _flags(df):
        """Бинарные признаки из текста strain и Culture."""
        text = (df.get("strain", "").astype(str) + " " +
                df.get("Culture", "").astype(str))
        cols = [text.str.contains(pat, case=False, regex=True, na=False).to_numpy(dtype=float)
                for pat in TEXT_FLAGS.values()]
        return np.column_stack(cols) if cols else np.zeros((len(df), 0))

    def _maybe_flags(self, df):
        return self._flags(df) if self.use_flags else np.zeros((len(df), 0))

    def _tables(self, df):
        """Статистики по группам одной обучающей выборки."""
        out = {}
        prior_s = df["is_sentinel"].mean()
        act = df["is_sentinel"] == 0
        prior_m = (np.log10(np.clip(df.loc[act, "y"], 1e-3, None)).mean()
                   if act.any() else 0.0)
        for c in self.cols:
            if c not in df:
                continue
            g = df.groupby(c)["is_sentinel"].agg(["size", "sum"])
            rate = _smoothed_rate(g["size"], g["sum"], prior_s, self.smoothing)

            sub = df[act]
            if len(sub):
                lg = np.log10(np.clip(sub["y"], 1e-3, None))
                gm = lg.groupby(sub[c]).agg(["size", "sum"])
                mean = _smoothed_rate(gm["size"], gm["sum"], prior_m, self.smoothing)
            else:
                mean = pd.Series(dtype=float)

            out[c] = {"rate": rate, "mean": mean, "count": g["size"]}
        self.prior_s_, self.prior_m_ = float(prior_s), float(prior_m)
        return out

    def fit(self, df):
        self.tables_ = self._tables(self._prepare(df))
        return self

    def fit_transform(self, df):
        """
        Для обучающей выборки кодировка считается вне фолда: каждая строка
        получает значения, рассчитанные без её участия. Без этого модель
        увидела бы собственный ответ, зашитый в признак.
        """
        d = self._prepare(df)
        self.tables_ = self._tables(d)
        X = np.full((len(df), len(self.cols) * 3), np.nan)
        folds = scaffold_kfold(df["smiles"].tolist(), self.inner_splits,
                               seed=self.seed + 300)
        for tr_idx, va_idx in folds:
            tables = self._tables(d.iloc[tr_idx])
            X[va_idx] = self._apply(d.iloc[va_idx], tables)
        return np.hstack([X, self._maybe_flags(d)])

    # ---------------------------------------------------------- применение

    def _apply(self, df, tables):
        parts = []
        for c in self.cols:
            t = tables.get(c)
            if t is None:
                parts.append(np.full((len(df), 3), np.nan))
                continue
            key = df[c] if c in df else pd.Series([None] * len(df))
            parts.append(np.column_stack([
                key.map(t["rate"]).fillna(self.prior_s_).to_numpy(dtype=float),
                key.map(t["mean"]).fillna(self.prior_m_).to_numpy(dtype=float),
                key.map(t["count"]).fillna(0).to_numpy(dtype=float),
            ]))
        return np.hstack(parts) if parts else np.zeros((len(df), 0))

    def transform(self, df):
        d = self._prepare(df)
        return np.hstack([self._apply(d, self.tables_), self._maybe_flags(d)])

    @property
    def names(self):
        out = []
        for c in self.cols:
            out += [f"meta_{c}_sentinel_rate", f"meta_{c}_mean_log_si", f"meta_{c}_count"]
        return out + [f"flag_{n}" for n in TEXT_FLAGS]
