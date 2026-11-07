#!/usr/bin/env python3
"""
Валидация и сборка внешнего датасета.

Вход:  data/raw/chembl_activities.csv, data/raw/pubchem_activities.csv
Выход: data/external.csv  со схемой, которую требует задание:
       smiles, IC50, CC50, SI, doi  (+ служебные колонки)

Что делает:
  1. Приводит единицы к мкМ (nM, M, mM, нг/мл и мкг/мл через молекулярную массу).
  2. Канонизирует SMILES через RDKit, убирает соли, отбрасывает невалидное.
  3. Спаривает IC50 и CC50 одного соединения ВНУТРИ одной публикации —
     в ChEMBL это разные записи, связать их иначе нельзя.
  4. Считает SI = CC50 / IC50.
  5. Помечает цензурированные измерения (standard_relation '>'): в ChEMBL это
     означает «активность не обнаружена вплоть до этой концентрации», то есть
     прямой аналог заглушки SI = 10000 в обучающей выборке организаторов.
  6. Снимает дубликаты и разрешает конфликты (медиана по повторам).
  7. Проверяет пересечение с test.csv и train.csv соревнования.

Запуск:
    python validate.py --chembl ../data/raw/chembl_activities.csv \
                       --pubchem ../data/raw/pubchem_activities.csv \
                       --test ../data/test.csv --train ../data/train.csv \
                       --out ../data/external.csv
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.SaltRemover import SaltRemover

RDLogger.DisableLog("rdApp.*")

# Значение-заглушка организаторов: IC50 не определён → вещество неактивно
SENTINEL_SI = 10000.0

# Разумные границы: вне их почти наверняка ошибка единиц
MIN_UM, MAX_UM = 1e-6, 1e7

_remover = SaltRemover()

# ChEMBL пишет единицы сокращённо (uM, nM), PubChem — словами
# (MICROMOLAR, NANOMOLAR). Без словесных вариантов все измерения
# из PubChem молча отбрасывались бы как «неизвестные единицы».
UNIT_TO_UM = {
    "um": 1.0, "µm": 1.0, "umol/l": 1.0, "10'-6m": 1.0, "micromolar": 1.0,
    "nm": 1e-3, "10'-9m": 1e-3, "nanomolar": 1e-3,
    "mm": 1e3, "10'-3m": 1e3, "millimolar": 1e3,
    "m": 1e6, "molar": 1e6,
    "pm": 1e-6, "picomolar": 1e-6,
}
MASS_UNITS = {"ug/ml", "µg/ml", "ug.ml-1", "mg/ml", "ng/ml"}
MASS_FACTOR = {"ug/ml": 1.0, "µg/ml": 1.0, "ug.ml-1": 1.0, "mg/ml": 1e3, "ng/ml": 1e-3}


def norm_unit(u):
    if not isinstance(u, str):
        return ""
    return u.strip().lower().replace(" ", "")


def to_micromolar(value, unit, mol_weight):
    """Приводит значение к мкМ. Возвращает (значение, причина_отказа)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, "нет значения"
    u = norm_unit(unit)
    if not u:
        return None, "нет единиц"
    if u in UNIT_TO_UM:
        return float(value) * UNIT_TO_UM[u], None
    if u in MASS_UNITS:
        if not mol_weight or mol_weight <= 0:
            return None, "масс. единицы без молекулярной массы"
        # мкМ = (мкг/мл) * 1000 / M
        return float(value) * MASS_FACTOR[u] * 1000.0 / mol_weight, None
    return None, f"неизвестные единицы: {unit}"


def canonical_smiles(smi):
    """Канонический SMILES без солей. None, если молекула не читается."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = _remover.StripMol(mol, dontRemoveEverything=True)
    except Exception:
        pass
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    # если осталось несколько фрагментов, берём самый крупный
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def mol_weight(smi):
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Descriptors.MolWt(mol) if mol else None


# ---------------------------------------------------------------- ChEMBL

IC50_TYPES = {"IC50", "EC50", "GI50"}
CC50_TYPES = {"CC50", "TC50", "TD50"}


def load_chembl(path):
    if not Path(path).exists():
        print(f"  {path} не найден, пропускаю", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  прочитано записей: {len(df)}", file=sys.stderr)

    df["smiles_canon"] = df["canonical_smiles"].map(canonical_smiles)
    bad = df["smiles_canon"].isna().sum()
    df = df[df["smiles_canon"].notna()].copy()
    print(f"  отброшено с нечитаемым SMILES: {bad}", file=sys.stderr)

    mw = {s: mol_weight(s) for s in df["smiles_canon"].unique()}
    df["mw"] = df["smiles_canon"].map(mw)

    conv = df.apply(
        lambda r: to_micromolar(r.get("standard_value"), r.get("standard_units"), r.get("mw")),
        axis=1,
    )
    df["value_um"] = [c[0] for c in conv]
    df["reject"] = [c[1] for c in conv]

    rejected = df["value_um"].isna()
    if rejected.any():
        print(f"  не сконвертировано: {rejected.sum()}", file=sys.stderr)
        for reason, n in df.loc[rejected, "reject"].value_counts().head(5).items():
            print(f"    {reason}: {n}", file=sys.stderr)
    df = df[df["value_um"].notna()].copy()

    out_of_range = ~df["value_um"].between(MIN_UM, MAX_UM)
    if out_of_range.any():
        print(f"  вне допустимого диапазона, отброшено: {out_of_range.sum()}", file=sys.stderr)
    df = df[~out_of_range].copy()

    rel = df.get("standard_relation", pd.Series("=", index=df.index)).fillna("=")
    df["censored"] = rel.astype(str).str.strip().isin([">", ">="])

    df["kind"] = np.where(
        df["standard_type"].isin(IC50_TYPES), "IC50",
        np.where(df["standard_type"].isin(CC50_TYPES), "CC50", None),
    )
    df = df[df["kind"].notna()].copy()
    df["source"] = "ChEMBL"
    df["pair_key"] = df["smiles_canon"] + "|" + df["document_chembl_id"].astype(str)
    return df


# --------------------------------------------------------------- PubChem


def load_pubchem(path):
    if not Path(path).exists():
        print(f"  {path} не найден, пропускаю", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  прочитано записей: {len(df)}", file=sys.stderr)
    if df.empty:
        return pd.DataFrame()

    df["smiles_canon"] = df["smiles"].map(canonical_smiles)
    df = df[df["smiles_canon"].notna()].copy()
    mw = {s: mol_weight(s) for s in df["smiles_canon"].unique()}
    df["mw"] = df["smiles_canon"].map(mw)

    rows = []
    for _, r in df.iterrows():
        for kind in ("IC50", "CC50"):
            raw = r.get(f"{kind}_raw")
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            m = re.search(r"-?\d+\.?\d*(?:[eE]-?\d+)?", str(raw))
            if not m:
                continue
            unit = r.get(f"{kind}_unit") or "um"  # PubChem чаще всего в мкМ
            val, _ = to_micromolar(float(m.group()), unit, r.get("mw"))
            if val is None or not (MIN_UM <= val <= MAX_UM):
                continue
            rows.append({
                "smiles_canon": r["smiles_canon"],
                "kind": kind,
                "value_um": val,
                "censored": str(raw).strip().startswith(">"),
                "doi": r.get("doi", ""),
                "source": "PubChem",
                "source_url": r.get("source_url", ""),
                "document_chembl_id": f"AID{r.get('aid')}",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pair_key"] = out["smiles_canon"] + "|" + out["document_chembl_id"].astype(str)
    print(f"  извлечено измерений: {len(out)}", file=sys.stderr)
    return out


# ----------------------------------------------------------------- сборка


def aggregate(df):
    """Медиана по повторам одного типа в одной публикации."""
    g = (
        df.groupby(["pair_key", "smiles_canon", "kind"], as_index=False)
        .agg(
            value_um=("value_um", "median"),
            n_measurements=("value_um", "size"),
            spread=("value_um", lambda s: (s.max() / s.min()) if s.min() > 0 and len(s) > 1 else 1.0),
            censored=("censored", "all"),
            doi=("doi", "first"),
            source=("source", "first"),
            source_url=("source_url", "first"),
        )
    )
    conflict = g["spread"] > 10
    if conflict.any():
        print(f"  повторы с расхождением более 10x: {conflict.sum()} (оставлена медиана)", file=sys.stderr)
    return g


def pair_and_compute(g):
    """Спаривает IC50 и CC50 внутри (молекула, публикация) и считает SI."""
    wide = g.pivot_table(
        index=["pair_key", "smiles_canon"],
        columns="kind",
        values=["value_um", "censored"],
        aggfunc="first",
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    meta = g.drop_duplicates("pair_key").set_index("pair_key")[["doi", "source", "source_url"]]
    wide = wide.join(meta, on="pair_key")

    wide = wide.rename(columns={"value_um_IC50": "IC50", "value_um_CC50": "CC50"})
    for c in ("IC50", "CC50", "censored_IC50", "censored_CC50"):
        if c not in wide:
            wide[c] = np.nan

    ic_cens = wide["censored_IC50"].fillna(False).astype(bool)

    # SI: обычный случай
    si = wide["CC50"] / wide["IC50"]
    # цензурированный IC50 ('>'), то есть активность не обнаружена → заглушка
    si = si.where(~ic_cens, SENTINEL_SI)
    # IC50 вовсе отсутствует, но цитотоксичность измерена → тоже заглушка,
    # по той же логике, что и в train организаторов
    si = si.where(~(wide["IC50"].isna() & wide["CC50"].notna()), SENTINEL_SI)
    wide["SI"] = si
    wide["is_sentinel"] = (wide["SI"] == SENTINEL_SI).astype(int)

    # нужен хотя бы один из показателей
    wide = wide[wide["SI"].notna()].copy()
    return wide


def resolve_duplicates(df):
    """Одна молекула может встретиться в нескольких публикациях. Берём медиану SI."""
    before = len(df)
    agg = (
        df.groupby("smiles_canon", as_index=False)
        .agg(
            IC50=("IC50", "median"),
            CC50=("CC50", "median"),
            SI=("SI", "median"),
            is_sentinel=("is_sentinel", "max"),
            n_sources=("pair_key", "nunique"),
            si_spread=("SI", lambda s: (s.max() / s.min()) if s.min() > 0 and len(s) > 1 else 1.0),
            doi=("doi", lambda s: "; ".join(sorted({x for x in s if isinstance(x, str) and x}))),
            source=("source", lambda s: "; ".join(sorted(set(s)))),
            source_url=("source_url", "first"),
        )
    )
    print(f"  молекул: {len(agg)} (из {before} пар молекула-публикация)", file=sys.stderr)
    conflict = agg["si_spread"] > 10
    if conflict.any():
        print(f"  молекул с расхождением SI более 10x между публикациями: {conflict.sum()}", file=sys.stderr)
    return agg


def leak_check(df, test_path, train_path):
    """Проверка пересечения с выборками соревнования."""
    for name, path in (("test", test_path), ("train", train_path)):
        if not path or not Path(path).exists():
            print(f"  {name}: файл не найден, проверка пропущена", file=sys.stderr)
            continue
        comp = pd.read_csv(path)
        canon = {canonical_smiles(s) for s in comp["SMILES"].dropna()}
        canon.discard(None)
        overlap = df["smiles_canon"].isin(canon)
        print(f"  пересечение с {name}: {overlap.sum()} молекул", file=sys.stderr)
        df[f"in_{name}"] = overlap.astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chembl", default="../data/raw/chembl_activities.csv")
    ap.add_argument("--pubchem", default="../data/raw/pubchem_activities.csv")
    ap.add_argument("--test", default="../data/test.csv")
    ap.add_argument("--train", default="../data/train.csv")
    ap.add_argument("--out", default="../data/external.csv")
    args = ap.parse_args()

    print("ChEMBL:", file=sys.stderr)
    chembl = load_chembl(args.chembl)
    print("PubChem:", file=sys.stderr)
    pubchem = load_pubchem(args.pubchem)

    frames = [f for f in (chembl, pubchem) if not f.empty]
    if not frames:
        print("Нет входных данных. Сначала запусти fetch_*.py", file=sys.stderr)
        return 1

    keep = ["pair_key", "smiles_canon", "kind", "value_um", "censored", "doi", "source", "source_url"]
    for f in frames:
        for c in keep:
            if c not in f:
                f[c] = "" if c in ("doi", "source", "source_url") else np.nan
    df = pd.concat([f[keep] for f in frames], ignore_index=True)
    print(f"\nВсего измерений: {len(df)}", file=sys.stderr)

    print("Агрегация повторов:", file=sys.stderr)
    g = aggregate(df)

    print("Спаривание IC50/CC50 и расчёт SI:", file=sys.stderr)
    paired = pair_and_compute(g)
    print(f"  получено записей с SI: {len(paired)}", file=sys.stderr)
    print(f"  из них заглушек (неактивные): {int(paired['is_sentinel'].sum())}", file=sys.stderr)

    print("Снятие дубликатов между публикациями:", file=sys.stderr)
    final = resolve_duplicates(paired)

    print("Проверка на утечку:", file=sys.stderr)
    final = leak_check(final, args.test, args.train)

    final = final.rename(columns={"smiles_canon": "smiles"})
    cols = ["smiles", "IC50", "CC50", "SI", "doi", "is_sentinel", "n_sources",
            "si_spread", "source", "source_url"]
    cols += [c for c in ("in_test", "in_train") if c in final]
    final = final[cols].sort_values("smiles").reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out, index=False)

    print(f"\nЗаписано {len(final)} молекул в {out}", file=sys.stderr)
    print(f"  заглушек SI={SENTINEL_SI:g}: {int(final['is_sentinel'].sum())}", file=sys.stderr)
    active = final[final["is_sentinel"] == 0]["SI"]
    if len(active):
        print(f"  активных: {len(active)}, медиана SI {active.median():.2f}", file=sys.stderr)
    if "in_test" in final:
        print(f"  ВНИМАНИЕ: {int(final['in_test'].sum())} молекул есть в test — "
              f"исключи их при локальной валидации", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
