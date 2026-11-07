"""
Признаки из колонки Title.

Единственная колонка, которую мы не использовали. Она содержит не случайный
идентификатор, а название соединения в том виде, как его записали авторы
публикации, и форма названия говорит о происхождении молекулы:

  слово (Econazole nitrate)  —  0,0% заглушек, 13 молекул в train
  название с пробелом        —  6,7%
  sid-код (sid103692656)     — 20,0%
  код (PAV-825)              — 33,3%
  серийный номер (11c)       — 31,7%

Смысл прямой: готовые препараты с собственными именами хорошо изучены
и активны, а серийные номера — это соединения из синтетических серий,
где большинство вариантов оказываются неактивными и IC50 для них
не измеряют.

Само название почти уникально (278 различных на 320 молекул), поэтому
кодировать его напрямую нельзя — это было бы запоминание. Берутся только
грубые производные: тип названия, наличие буквенного суффикса, длина,
состав символов. Они переносятся на новые молекулы.
"""

import re

import numpy as np

SERIES = re.compile(r"^\d+[a-z]?$", re.I)
SID = re.compile(r"^sid\d+$", re.I)
CHEMBL = re.compile(r"^chembl\d+$", re.I)
CODE = re.compile(r"^[A-Z]{2,}-?\d+$")
SERIES_SUFFIX = re.compile(r"^(\d+)([a-z])$", re.I)


def title_kind(t):
    t = str(t).strip()
    if SID.match(t):
        return 0
    if CHEMBL.match(t):
        return 1
    if SERIES.match(t):
        return 2
    if CODE.match(t):
        return 3
    if " " in t:
        return 4
    if t.isalpha():
        return 5
    return 6


N_KINDS = 7
NAMES = (
    [f"title_kind_{i}" for i in range(N_KINDS)]
    + ["title_len", "title_n_digits", "title_n_alpha", "title_n_dash",
       "title_has_suffix", "title_series_num", "title_digit_frac",
       "title_is_long_name"]
)


def build_title(titles):
    """Грубые производные названия. Само название не кодируется."""
    rows = []
    for t in titles:
        t = str(t).strip()
        k = title_kind(t)
        onehot = [1.0 if i == k else 0.0 for i in range(N_KINDS)]
        n_dig = sum(c.isdigit() for c in t)
        n_alp = sum(c.isalpha() for c in t)
        m = SERIES_SUFFIX.match(t)
        rows.append(onehot + [
            float(len(t)),
            float(n_dig),
            float(n_alp),
            float(t.count("-")),
            1.0 if m else 0.0,
            float(m.group(1)) if m else -1.0,
            n_dig / max(len(t), 1),
            # длинное название с буквами и без цифр — обычно
            # систематическое химическое имя или название препарата
            1.0 if (len(t) > 12 and n_alp > n_dig) else 0.0,
        ])
    return np.asarray(rows, dtype=float), list(NAMES)
