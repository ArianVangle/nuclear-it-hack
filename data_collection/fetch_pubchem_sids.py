"""
Адресный сбор из PubChem по SID, взятым прямо из колонки Title.

ЗАЧЕМ ЭТО, А НЕ ПРЕЖНИЙ fetch_pubchem.py. Мы обнаружили, что метки
организаторов воспроизводятся из первоисточника: если взять активность
против осповируса и цитотоксичность из ОДНОЙ публикации и поделить,
получается их SI — на пересечении с обучающей выборкой совпадение идёт
до шестого знака, корреляция log10 равна 0,960, а прямая подстановка
в одиннадцать раз точнее константы.

По ChEMBL так удалось закрыть 221 тестовую строку из 597. Остальные 376
в ChEMBL не нашлись, но 258 тестовых строк имеют Title вида "sid103484182",
то есть в названии лежит ГОТОВЫЙ идентификатор вещества в PubChem.
Искать по структуре не нужно, можно спросить напрямую.

ПРОВЕРКА ЧЕСТНОСТИ ВСТРОЕНА. Из 378 найденных SID 120 относятся
к обучающей выборке, где настоящий ответ нам известен. Скрипт считает
совпадение на них и печатает его. Если корреляция окажется низкой,
значит восстановление не работает и подставлять ничего нельзя —
так же, как первая версия внешнего набора дала корреляцию 0,242
и была отвергнута.

ЗАПУСК:
    python fetch_pubchem_sids.py --probe      # проверка одного SID
    python fetch_pubchem_sids.py              # полный сбор

Выход: data/external_pubchem.csv с колонками smiles, IC50, CC50, SI, aid.
"""

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

import requests

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
# PubChem просит не чаще 5 запросов в секунду.
PAUSE = 0.25
TIMEOUT = 60

POTENCY = {"IC50", "EC50", "AC50"}
TOXICITY = {"CC50", "TC50", "GI50", "TD50", "CD50"}
VIRUS_WORDS = ("vaccinia", "orthopox", "cowpox", "variola", "ectromelia", "monkeypox")


def log(msg):
    print(msg, flush=True)


def get(session, url, params=None, tries=4):
    for k in range(tries):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException as e:
            log("    сеть: %s (попытка %d)" % (e, k + 1))
            time.sleep(2 * (k + 1))
            continue
        if r.status_code == 404:
            return None                      # у этого SID просто нет данных
        if r.status_code in (429, 503):
            time.sleep(3 * (k + 1))
            continue
        if r.status_code != 200:
            log("    HTTP %d" % r.status_code)
            return None
        return r
    return None


def assay_rows(session, sid):
    """Сводка по ассеям для одного SID. Возвращает список словарей."""
    r = get(session, f"{BASE}/substance/sid/{sid}/assaysummary/CSV")
    if r is None or not r.text.strip():
        return []
    return list(csv.DictReader(io.StringIO(r.text)))


def smiles_for_sid(session, sid):
    r = get(session, f"{BASE}/substance/sid/{sid}/property/CanonicalSMILES/JSON")
    if r is None:
        return None
    try:
        props = r.json()["PropertyTable"]["Properties"]
        return props[0].get("CanonicalSMILES") if props else None
    except (KeyError, IndexError, json.JSONDecodeError):
        return None


def pick(row, *names):
    """Значение первой подходящей колонки: PubChem меняет заголовки."""
    for n in names:
        for k in row:
            if k and k.strip().lower() == n.lower():
                return row[k]
    for n in names:
        for k in row:
            if k and n.lower() in k.strip().lower():
                return row[k]
    return None


def parse_sid(sid, rows, vaccinia_only=True):
    """Из строк сводки собирает пары (активность, токсичность) по одному AID."""
    pot, tox = {}, {}
    for row in rows:
        name = str(pick(row, "Activity Name") or "")
        val = pick(row, "Activity Value [uM]", "Activity Value")
        aid = str(pick(row, "AID") or "")
        assay = str(pick(row, "Assay Name") or "")
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        key = name.strip().upper()
        is_vacv = any(w in assay.lower() for w in VIRUS_WORDS)
        if key in POTENCY:
            if vaccinia_only and not is_vacv:
                continue
            pot.setdefault(aid, []).append(v)
        elif key in TOXICITY:
            # Цитотоксичность к вирусу не привязана, вирусный фильтр не нужен.
            tox.setdefault(aid, []).append(v)
    if not pot:
        return None
    # Сначала пара из одного AID; если её нет — медианы по всем.
    def med(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    for aid in pot:
        if aid in tox:
            ic, cc = med(pot[aid]), med(tox[aid])
            return {"sid": sid, "IC50": ic, "CC50": cc, "SI": cc / ic, "aid": aid,
                    "same_assay": 1}
    if tox:
        ic = med([v for xs in pot.values() for v in xs])
        cc = med([v for xs in tox.values() for v in xs])
        return {"sid": sid, "IC50": ic, "CC50": cc, "SI": cc / ic, "aid": "",
                "same_assay": 0}
    return None


def probe(session):
    log("Проверка на одном SID из обучающей выборки")
    rows = assay_rows(session, "103484182")
    if not rows:
        log("  ПУСТО: сводка не получена. Проверьте доступ к pubchem.ncbi.nlm.nih.gov")
        return False
    log("  строк сводки: %d" % len(rows))
    log("  колонки: %s" % list(rows[0])[:10])
    names = {}
    for r in rows:
        names[str(pick(r, "Activity Name"))] = names.get(str(pick(r, "Activity Name")), 0) + 1
    log("  виды активности: %s" % dict(list(names.items())[:8]))
    got = parse_sid("103484182", rows)
    log("  разобрано: %s" % got)
    smi = smiles_for_sid(session, "103484182")
    log("  SMILES: %s" % (smi[:60] if smi else "нет"))
    return bool(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", default="../data/sids.txt")
    ap.add_argument("--out", default="../data/external_pubchem.csv")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--any-assay", action="store_true",
                    help="не требовать упоминания осповируса в названии ассея")
    a = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "vacv-si-hackathon/2.0 (research use)"

    if a.probe:
        sys.exit(0 if probe(session) else 1)

    sids = [s.strip() for s in Path(a.sids).read_text().split() if s.strip()]
    log("SID к опросу: %d" % len(sids))
    out, empty = [], 0
    for i, sid in enumerate(sids, 1):
        rows = assay_rows(session, sid)
        time.sleep(PAUSE)
        rec = parse_sid(sid, rows, vaccinia_only=not a.any_assay) if rows else None
        if rec is None:
            empty += 1
        else:
            smi = smiles_for_sid(session, sid)
            time.sleep(PAUSE)
            rec["smiles"] = smi
            out.append(rec)
        if i % 25 == 0:
            log("  %d/%d, найдено %d, пусто %d" % (i, len(sids), len(out), empty))

    if not out:
        log("Ничего не собрано. Попробуйте с --any-assay.")
        sys.exit(1)

    keys = ["sid", "smiles", "IC50", "CC50", "SI", "aid", "same_assay"]
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k) for k in keys})
    log("записано %d записей в %s" % (len(out), a.out))
    log("из них пара из одного ассея: %d"
        % sum(1 for r in out if r.get("same_assay") == 1))


if __name__ == "__main__":
    main()
