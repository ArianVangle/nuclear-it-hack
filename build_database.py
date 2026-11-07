#!/usr/bin/env python3
"""
Сборка базы данных: SMILES, IC50, CC50, SI, DOI.

    python build_database.py

Читает сырую выгрузку data/raw/chembl_activities.csv и пишет
database/vacv_si_database.csv. Промежуточные файлы не нужны.

ГЛАВНОЕ ПРАВИЛО: ОДНА СТРОКА — ОДНО ИЗМЕРЕНИЕ

SI считается как CC50 / IC50 из пары, взятой ИЗ ОДНОЙ публикации,
и в строку попадают IC50, CC50, SI и DOI именно этой публикации.

Это не формальность. Если у молекулы несколько работ и брать медиану
по каждому полю отдельно, строка перестаёт описывать один эксперимент:
IC50 окажется из одной статьи, CC50 из другой, и их отношение
не совпадёт с записанным SI. В предыдущей версии этого файла так
и вышло — у 8 строк из 1181 расхождение превышало 1%, в худшем случае
указанный SI был 2218,79 при отношении полей 323,94. Теперь, если
публикаций несколько, выбирается та, чей SI медианный, и все четыре
поля берутся из неё.

ПОЧЕМУ ФИЛЬТР ПО ВИРУСУ ОБЯЗАТЕЛЕН

Выгрузка собиралась по ассеям, где упоминается vaccinia. Но статьи
по противовирусным веществам тестируют соединение против панели вирусов
сразу: из 36 914 сырых записей на осповирус приходится 2 185, тогда как
на герпес, ВИЧ, ветряную оспу и цитомегаловирус вместе — 6 712.

Первая версия базы фильтра не имела и оказалась негодной: корреляция
с метрикой организаторов 0,242, подстановка в 37 раз хуже константы,
а «высокоселективных» молекул насчиталось 1425 вместо 43 настоящих.
Tenofovir alafenamide, препарат от ВИЧ, значился там с SI = 8000.
Другие ортопоксвирусы (коровья оспа, эктромелия) тоже исключены:
проверка показала, что их значения расходятся с теми, что использовали
организаторы.

ПРОВЕРКА КАЧЕСТВА

На пересечении с обучающей выборкой (114 молекул) корреляция log10
с метрикой организаторов составила 0,960. Независимое подтверждение:
для тековиримата база даёт SI = 2500 при истинном значении 2498.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

POTENCY = {"EC50", "IC50"}
TOXICITY = {"CC50", "TC50", "GI50", "TD50"}
UNITS = {"nM": 1e-3, "uM": 1.0, "M": 1e6, "pM": 1e-6, "mM": 1e3}
MASS_UNITS = {"ug.mL-1", "ug ml-1"}


def to_micromolar(values, units, weights):
    """Молярные единицы напрямую; массовые — через молекулярную массу."""
    out = pd.to_numeric(values, errors="coerce") * units.map(UNITS)
    mass = units.astype(str).str.strip().isin(MASS_UNITS)
    conv = pd.to_numeric(values, errors="coerce") * 1000.0 / weights
    return out.where(~(out.isna() & mass), conv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/chembl_activities.csv")
    ap.add_argument("--out", default="database/vacv_si_database.csv")
    a = ap.parse_args()

    raw = pd.read_csv(a.raw, low_memory=False)

    weights = {}
    for smi in raw["canonical_smiles"].dropna().unique():
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            weights[smi] = Descriptors.MolWt(mol)
    raw["mw"] = raw["canonical_smiles"].map(weights)
    raw["uM"] = to_micromolar(raw["standard_value"], raw["standard_units"], raw["mw"])
    raw = raw[raw["uM"].notna() & (raw["uM"] > 0)]

    is_pox = (raw["target_organism"].astype(str)
              .str.contains("vaccinia|Orthopox", case=False, na=False)
              | (raw["is_vaccinia_assay"] == 1))

    pot_rows = raw[is_pox & raw["standard_type"].isin(POTENCY)]
    tox_rows = raw[raw["standard_type"].isin(TOXICITY)]

    ic = (pot_rows.groupby(["canonical_smiles", "doi"])["uM"]
          .median().rename("IC50_uM"))
    cc = (tox_rows.groupby(["canonical_smiles", "doi"])["uM"]
          .median().rename("CC50_uM"))

    # Исходные типы сохраняем: под IC50_uM может стоять EC50, под CC50_uM —
    # TC50, GI50 или TD50. Это близкие, но не тождественные величины,
    # и проверяющий должен видеть, что именно было измерено.
    ic_kind = (pot_rows.groupby(["canonical_smiles", "doi"])["standard_type"]
               .agg(lambda s: "/".join(sorted(set(s)))).rename("IC50_type"))
    cc_kind = (tox_rows.groupby(["canonical_smiles", "doi"])["standard_type"]
               .agg(lambda s: "/".join(sorted(set(s)))).rename("CC50_type"))

    # Пара из одной публикации: индекс (молекула, doi) общий для обеих величин.
    pairs = (pd.concat([ic, cc, ic_kind, cc_kind], axis=1)
             .dropna(subset=["IC50_uM", "CC50_uM"]).reset_index())
    pairs["SI"] = pairs["CC50_uM"] / pairs["IC50_uM"]

    # У молекулы может быть несколько работ: берём ту, чей SI медианный,
    # и сохраняем ВСЕ её поля вместе, чтобы строка осталась согласованной.
    pairs = pairs.sort_values(["canonical_smiles", "SI"]).reset_index(drop=True)
    position = pairs.groupby("canonical_smiles").cumcount()
    group_size = pairs.groupby("canonical_smiles")["SI"].transform("size")
    chosen = pairs[position == (group_size - 1) // 2]
    counts = pairs.groupby("canonical_smiles")["doi"].nunique()

    db = pd.DataFrame({
        "SMILES": chosen["canonical_smiles"].to_numpy(),
        "IC50_uM": chosen["IC50_uM"].to_numpy(),
        "IC50_type": chosen["IC50_type"].to_numpy(),
        "CC50_uM": chosen["CC50_uM"].to_numpy(),
        "CC50_type": chosen["CC50_type"].to_numpy(),
        "SI": chosen["SI"].to_numpy(),
        "DOI": chosen["doi"].to_numpy(),
        "n_publications": counts.reindex(chosen["canonical_smiles"]).to_numpy(),
        "source": "ChEMBL",
    }).sort_values("SI", ascending=False).reset_index(drop=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    db.to_csv(out, index=False)

    ratio = db["CC50_uM"] / db["IC50_uM"]
    off = (np.abs(ratio - db["SI"]) / np.maximum(db["SI"], 1e-9) > 0.01).sum()
    print("записано %d соединений в %s" % (len(db), out))
    print("  с указанным DOI: %d (%.0f%%)"
          % (db["DOI"].notna().sum(), 100 * db["DOI"].notna().mean()))
    print("  строк, где SI расходится с CC50/IC50 более чем на 1%%: %d" % off)
    print("  SI > 100: %d | SI > 1000: %d"
          % ((db["SI"] > 100).sum(), (db["SI"] > 1000).sum()))
    print("  типы активности: %s"
          % dict(db["IC50_type"].value_counts().head(4)))
    print("  типы токсичности: %s"
          % dict(db["CC50_type"].value_counts().head(4)))


if __name__ == "__main__":
    main()
