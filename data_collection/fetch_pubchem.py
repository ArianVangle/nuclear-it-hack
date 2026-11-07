#!/usr/bin/env python3
"""
Выгрузка биоскринингов против вируса осповакцины из PubChem BioAssay.

Источники:
  - NCBI E-utilities (поиск AID):  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pcassay
  - PubChem PUG REST (данные):     https://pubchem.ncbi.nlm.nih.gov/rest/pug

Предупреждение: данные PubChem BioAssay гораздо менее структурированы, чем ChEMBL.
Названия колонок с величинами задаёт автор скрининга, единой схемы нет. Скрипт
делает эвристическое извлечение и сохраняет сырые CSV, чтобы можно было
проверить вручную. Считай его источником кандидатов, а не готовых данных.

PubChem просит не более 5 запросов в секунду.

Запуск:
    python fetch_pubchem.py --out ../data/raw/pubchem_activities.csv
"""

import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 90
RETRIES = 4
SLEEP = 0.25  # ≤ 5 запросов в секунду

QUERY = "vaccinia"
MAX_AIDS = 300

# Как распознать колонку с величиной
IC50_PAT = re.compile(r"\b(ic50|ec50|ic_50|ec_50)\b", re.I)
CC50_PAT = re.compile(r"\b(cc50|tc50|td50|cytotox\w*)\b", re.I)
SI_PAT = re.compile(r"\b(si|selectivity[ _]?index)\b", re.I)
UNIT_PAT = re.compile(r"\((u|µ|n|m)m\)|\b(um|µm|nm|mm|ug/?ml|µg/?ml)\b", re.I)


def get(session, url, params=None, as_text=False):
    last = None
    for attempt in range(RETRIES):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.text if as_text else r.json()
            if r.status_code == 404:
                return None  # у многих AID нет нужного раздела — это норма
            last = f"HTTP {r.status_code}"
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(2 ** attempt)
    print(f"  пропуск {url}: {last}", file=sys.stderr)
    return None


def search_aids(session, query, retmax):
    """Поиск AID скринингов по тексту через E-utilities."""
    data = get(
        session,
        f"{EUTILS}/esearch.fcgi",
        {"db": "pcassay", "term": query, "retmax": retmax, "retmode": "json"},
    )
    if not data:
        return []
    return data.get("esearchresult", {}).get("idlist", [])


def assay_description(session, aid):
    data = get(session, f"{PUG}/assay/aid/{aid}/description/JSON")
    if not data:
        return {}
    try:
        desc = data["PC_AssayContainer"][0]["assay"]["descr"]
    except (KeyError, IndexError, TypeError):
        return {}
    name = desc.get("name", "")
    comment = " ".join(desc.get("comment", []) or [])
    description = " ".join(desc.get("description", []) or [])
    # DOI иногда лежит в ссылках на публикацию
    doi = ""
    for xref in desc.get("xref", []) or []:
        x = xref.get("xref", {})
        if "doi" in x:
            doi = x["doi"]
            break
    pmid = ""
    for xref in desc.get("xref", []) or []:
        x = xref.get("xref", {})
        if "pmid" in x:
            pmid = str(x["pmid"])
            break
    return {
        "name": name,
        "comment": comment[:2000],
        "description": description[:2000],
        "doi": doi,
        "pmid": pmid,
    }


# PubChem вставляет после заголовка служебные строки, описывающие типы,
# подписи и единицы колонок. Их легко принять за измерения: значение
# выглядит как "FLOAT" или "uM", а не как число.
META_TAGS = {"RESULT_TYPE", "RESULT_DESCR", "RESULT_UNIT", "RESULT_IS_ACTIVE_CONCENTRATION"}


def assay_table(session, aid):
    """CSV скрининга. Возвращает (строки_данных, единицы_по_колонкам)."""
    text = get(session, f"{PUG}/assay/aid/{aid}/CSV", as_text=True)
    if not text:
        return None, {}
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except csv.Error:
        return None, {}

    units = {}
    data = []
    for r in rows:
        tag = (r.get("PUBCHEM_RESULT_TAG") or "").strip().upper()
        if tag == "RESULT_UNIT":
            units = {k: (v or "").strip() for k, v in r.items() if (v or "").strip()}
        if tag in META_TAGS:
            continue
        data.append(r)
    return data, units


SMILES_COLS = ("PUBCHEM_EXT_DATASOURCE_SMILES", "PUBCHEM_SMILES", "SMILES")


def row_smiles(row):
    """SMILES прямо из таблицы скрининга — лишний запрос к API не нужен."""
    for c in SMILES_COLS:
        v = (row.get(c) or "").strip()
        if v:
            return v
    return ""


def cid_to_smiles(session, cids):
    """CID → SMILES, пачками. Запасной путь, если в таблице SMILES нет.

    Имя свойства в PUG REST менялось, поэтому пробуем несколько.
    """
    out = {}
    cids = [c for c in cids if c]
    CHUNK = 100
    for i in range(0, len(cids), CHUNK):
        chunk = cids[i : i + CHUNK]
        for prop in ("SMILES", "CanonicalSMILES", "ConnectivitySMILES"):
            data = get(
                session,
                f"{PUG}/compound/cid/{','.join(map(str, chunk))}/property/{prop}/JSON",
            )
            if data:
                got = False
                for p in data.get("PropertyTable", {}).get("Properties", []):
                    for k in (prop, "SMILES", "CanonicalSMILES", "ConnectivitySMILES"):
                        if k in p:
                            out[str(p["CID"])] = p[k]
                            got = True
                            break
                if got:
                    break
            time.sleep(SLEEP)
        time.sleep(SLEEP)
    return out


# Вирусы, упоминание которых в НАЗВАНИИ скрининга означает, что он не про оспу.
OTHER_VIRUSES = re.compile(
    r"ebola|ebov|marburg|\bhiv\b|influenza|hepatitis|\bhcv\b|\bhbv\b|dengue|zika|"
    r"sars|corona|\bcov\b|herpes|\bhsv\b|\bcmv\b|\brsv\b|rhinovirus|enterovirus|"
    r"chikungunya|west nile|yellow fever|rabies|measles|norovirus|rotavirus|polio",
    re.I)
VACCINIA = re.compile(r"vaccinia|\bVACV\b|\bVV\b|cowpox|monkeypox|variola|orthopox", re.I)


def is_relevant(desc):
    """
    Текстовый поиск по pcassay находит патенты, перечисляющие десятки вирусов:
    первый же результат по запросу vaccinia оказался скринингом против Эболы
    с ремдесивиром в качестве соединения. Слово «vaccinia» нашлось в описании
    патента, но скрининг не про оспу.

    Поэтому решает НАЗВАНИЕ скрининга, а не всё описание: если в названии
    указан другой вирус, скрининг отбрасывается, даже когда оспа где-то
    упомянута. Описание учитывается, только если название не называет
    никакого вируса вовсе.
    """
    name = desc.get("name", "") or ""
    if VACCINIA.search(name):
        return True
    if OTHER_VIRUSES.search(name):
        return False
    body = " ".join([desc.get("description", ""), desc.get("comment", "")])
    return bool(VACCINIA.search(body))


def classify_columns(fieldnames):
    """Эвристика: какие колонки таблицы содержат IC50, CC50, SI."""
    found = {"ic50": [], "cc50": [], "si": []}
    for col in fieldnames or []:
        if CC50_PAT.search(col):
            found["cc50"].append(col)
        elif IC50_PAT.search(col):
            found["ic50"].append(col)
        elif SI_PAT.search(col) and len(col) < 30:
            found["si"].append(col)
    return found


def guess_unit(col):
    m = UNIT_PAT.search(col)
    return m.group(0).strip("()") if m else ""


def probe(session, aid=None):
    """Разбор одного скрининга с печатью всего, что получаем."""
    if aid is None:
        aids = search_aids(session, QUERY, 5)
        print(f"поиск вернул AID: {aids}", file=sys.stderr)
        if not aids:
            print("  поиск не дал результатов", file=sys.stderr)
            return 1
        aid = aids[0]
    print(f"\nразбираем AID {aid}", file=sys.stderr)

    url = f"{PUG}/assay/aid/{aid}/description/JSON"
    print(f"  GET {url}", file=sys.stderr)
    desc = assay_description(session, aid)
    print(f"  описание: name={desc.get('name','')[:70]!r} doi={desc.get('doi','')!r}",
          file=sys.stderr)

    url = f"{PUG}/assay/aid/{aid}/CSV"
    print(f"  GET {url}", file=sys.stderr)
    text = get(session, url, as_text=True)
    if text is None:
        print("  ТАБЛИЦА НЕ ПОЛУЧЕНА (404 или ошибка) — это и есть причина нуля строк",
              file=sys.stderr)
        return 1
    print(f"  получено {len(text)} байт", file=sys.stderr)
    print(f"  про vaccinia: {is_relevant(desc)}", file=sys.stderr)
    table, units = assay_table(session, aid)
    if not table:
        print("  таблица пуста или не разобралась", file=sys.stderr)
        print(f"  первые 300 символов: {text[:300]!r}", file=sys.stderr)
        return 1
    print(f"  строк данных (без служебных): {len(table)}", file=sys.stderr)
    print(f"  единицы из метастроки: {units or '—'}", file=sys.stderr)
    print(f"  SMILES в таблице у первой строки: {row_smiles(table[0])[:60]!r}",
          file=sys.stderr)
    print(f"  КОЛОНКИ: {list(table[0].keys())}", file=sys.stderr)
    cols = classify_columns(table[0].keys())
    print(f"  распознано как IC50: {cols['ic50']}", file=sys.stderr)
    print(f"  распознано как CC50: {cols['cc50']}", file=sys.stderr)
    print(f"  распознано как SI:   {cols['si']}", file=sys.stderr)
    if not (cols["ic50"] or cols["cc50"]):
        print("  НИ ОДНА КОЛОНКА НЕ РАСПОЗНАНА — вот причина нуля строк.", file=sys.stderr)
        print("  Пришли список колонок выше, подправлю регулярные выражения.",
              file=sys.stderr)
        return 1
    print("  пример первой строки:", file=sys.stderr)
    for k, v in list(table[0].items())[:12]:
        print(f"    {k} = {v!r}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="разобрать один скрининг и выйти")
    ap.add_argument("--aid", help="конкретный AID для --probe")
    ap.add_argument("--out", default="../data/raw/pubchem_activities.csv")
    ap.add_argument("--raw-dir", default="../data/raw/pubchem_assays")
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--max-aids", type=int, default=MAX_AIDS)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "vacv-si-hackathon/1.0 (research use)"

    if args.probe:
        return probe(session, args.aid)

    print(f"Поиск скринингов по запросу «{args.query}»", file=sys.stderr)
    aids = search_aids(session, args.query, args.max_aids)
    print(f"  найдено AID: {len(aids)}", file=sys.stderr)
    if not aids:
        return 1

    rows = []
    from collections import Counter
    stats = Counter()
    unknown_cols = Counter()
    for n, aid in enumerate(aids, 1):
        print(f"  [{n}/{len(aids)}] AID {aid}", file=sys.stderr)
        desc = assay_description(session, aid)
        time.sleep(SLEEP)
        if not is_relevant(desc):
            stats["не про vaccinia"] += 1
            continue
        table, units = assay_table(session, aid)
        time.sleep(SLEEP)
        if not table:
            stats["нет таблицы или только служебные строки"] += 1
            continue

        # сохраняем сырьё для ручной проверки
        with (raw_dir / f"aid_{aid}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=table[0].keys())
            w.writeheader()
            w.writerows(table)

        cols = classify_columns(table[0].keys())
        if not (cols["ic50"] or cols["cc50"]):
            stats["колонки не распознаны"] += 1
            unknown_cols.update(table[0].keys())
            continue

        # SMILES обычно есть прямо в таблице; API дёргаем только для остатка
        missing = {r.get("PUBCHEM_CID") for r in table if not row_smiles(r)}
        smiles_map = cid_to_smiles(session, missing) if missing else {}

        n_before = len(rows)
        for r in table:
            cid = r.get("PUBCHEM_CID")
            smi = row_smiles(r) or smiles_map.get(str(cid), "")
            if not smi:
                stats["нет SMILES"] += 1
                continue
            rec = {
                "aid": aid,
                "cid": cid,
                "smiles": smi,
                "assay_name": desc.get("name", ""),
                "doi": desc.get("doi", ""),
                "pmid": desc.get("pmid", ""),
                "source_url": f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
            }
            wrote = False
            for kind, key in (("ic50", "IC50"), ("cc50", "CC50"), ("si", "SI")):
                for col in cols[kind]:
                    val = (r.get(col) or "").strip()
                    if not val:
                        continue
                    rec[f"{key}_raw"] = val
                    rec[f"{key}_col"] = col
                    rec[f"{key}_unit"] = units.get(col) or guess_unit(col)
                    wrote = True
                    break
            if wrote:
                rows.append(rec)
            else:
                stats["нет значений в распознанных колонках"] += 1
        if len(rows) == n_before:
            stats["скрининг не дал ни одной строки"] += 1

    cols_out = [
        "aid", "cid", "smiles", "assay_name", "doi", "pmid", "source_url",
        "IC50_raw", "IC50_col", "IC50_unit",
        "CC50_raw", "CC50_col", "CC50_unit",
        "SI_raw", "SI_col", "SI_unit",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols_out, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Записано {len(rows)} строк в {out}", file=sys.stderr)
    if stats:
        print("Почему скрининги отброшены:", file=sys.stderr)
        for k, v in stats.most_common():
            print(f"  {k}: {v}", file=sys.stderr)
    if not rows and unknown_cols:
        print("Чаще всего встречались колонки (подправь регулярные выражения):",
              file=sys.stderr)
        for c, v in unknown_cols.most_common(25):
            print(f"  {v:4d}  {c}", file=sys.stderr)
    print(f"Сырые таблицы скринингов: {raw_dir}", file=sys.stderr)
    print("Проверь вручную: названия колонок и единицы PubChem не стандартизированы.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
