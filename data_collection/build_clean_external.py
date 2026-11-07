"""
Пересборка внешнего набора из сырых записей ChEMBL.

ЗАЧЕМ. Первый вариант (data/external.csv) брал колонку SI как она есть,
и это оказалось ошибкой, которая стоила нам половины дня.

Сбор искал ассеи, где организм или описание УПОМИНАЕТ vaccinia. Но статьи
по противовирусным веществам почти всегда тестируют соединение против
панели вирусов сразу. В итоге из 36 914 сырых записей на осповирус
приходится 2 185, а остальное — герпес (2 198), ВИЧ (1 897), ветряная
оспа (1 339), цитомегаловирус (1 278) и цитотоксичность на человеке
(6 810). Селективность против ВИЧ попадала к нам как «высокий SI».

Что это давало на практике. В старом файле 1425 молекул с SI > 500.
На обучающей выборке соревнования у молекул с внешним SI выше 2000
медиана настоящего SI равна 0,43 и 44% нулей — то есть мы учили
классификатор хвоста на худшей из возможных групп. Tenofovir alafenamide
(препарат от ВИЧ и гепатита B) имел внешний SI 8000, и модель поднимала
ему прогноз против осповируса до 2070.

Вдобавок три колонки старого файла не согласованы между собой: отношение
CC50/IC50 воспроизводит собственную колонку SI с AUC 0,468, потому что
значения склеивались по молекуле из разных измерений.

ЧТО ДЕЛАЕТ ЭТОТ СКРИПТ. Считает SI сам, и только из пар, где обе
величины взяты из ОДНОЙ публикации:

  активность  — EC50 или IC50, мишень Orthopoxvirus vaccinia
                (или флаг is_vaccinia_assay)
  токсичность — CC50, TC50, GI50 или TD50 в той же публикации

Выход: 1078 молекул, из них 43 с SI > 500. Верхушка — бринцидофовир,
тековиримат и его аналоги, алкоксиалкильные эфиры цидофовира, то есть
настоящие противооспенные вещества.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POTENCY = {"EC50", "IC50"}
TOXICITY = {"CC50", "TC50", "GI50", "TD50"}

# В мкМ. Массовые единицы (ug.mL-1) пересчитать нельзя без молярной массы,
# поэтому такие записи отбрасываются, а не угадываются.
UNITS = {"nM": 1e-3, "uM": 1.0, "M": 1e6, "pM": 1e-6, "mM": 1e3}


def to_micromolar(df):
    v = pd.to_numeric(df["standard_value"], errors="coerce")
    return v * df["standard_units"].map(UNITS)


def build(raw_path):
    r = pd.read_csv(raw_path, low_memory=False)
    r["uM"] = to_micromolar(r)
    r = r[r["uM"].notna() & (r["uM"] > 0)]

    is_vacv = (
        r["target_organism"].astype(str).str.contains("vaccinia|Orthopox",
                                                      case=False, na=False)
        | (r["is_vaccinia_assay"] == 1)
    )
    potency = r[is_vacv & r["standard_type"].isin(POTENCY)]
    # Токсичность берём без требования к мишени: это измерение на клетках,
    # а привязку обеспечивает совпадение публикации.
    toxicity = r[r["standard_type"].isin(TOXICITY)]

    ic = potency.groupby(["canonical_smiles", "doi"])["uM"].median().rename("IC50")
    cc = toxicity.groupby(["canonical_smiles", "doi"])["uM"].median().rename("CC50")
    pair = pd.concat([ic, cc], axis=1).dropna().reset_index()
    pair["SI"] = pair["CC50"] / pair["IC50"]

    # Одна молекула может встречаться в нескольких публикациях: берём медиану
    # и запоминаем разброс, он пригодится как мера доверия.
    out = (pair.groupby("canonical_smiles")
                .agg(IC50=("IC50", "median"), CC50=("CC50", "median"),
                     SI=("SI", "median"), si_min=("SI", "min"),
                     si_max=("SI", "max"), n_papers=("doi", "nunique"),
                     doi=("doi", lambda s: "; ".join(sorted(set(s))[:3])))
                .reset_index()
                .rename(columns={"canonical_smiles": "smiles"}))
    out["si_spread"] = out["si_max"] / np.maximum(out["si_min"], 1e-9)
    return out.sort_values("SI", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/raw/chembl_activities.csv")
    ap.add_argument("--out", default="../data/external_clean.csv")
    a = ap.parse_args()
    df = build(Path(a.raw))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print("молекул: %d" % len(df))
    print("SI>100: %d | >500: %d | >1000: %d"
          % ((df.SI > 100).sum(), (df.SI > 500).sum(), (df.SI > 1000).sum()))
    print("записано в", a.out)


if __name__ == "__main__":
    main()
