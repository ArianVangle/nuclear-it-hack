#!/usr/bin/env python3
"""
Выгрузка данных об активности против вируса осповакцины из ChEMBL.

Источник: ChEMBL REST API, https://www.ebi.ac.uk/chembl/api/data
Документация: https://www.ebi.ac.uk/chembl/api/data/docs

Логика в три шага:

1. Находим ассеи (assay), у которых организм или описание упоминает vaccinia.
2. Собираем активности из тех же публикаций (document). Это принципиально:
   CC50 (цитотоксичность) и IC50 (ингибирование) для одного соединения лежат
   в разных записях ChEMBL и связываются только через общую статью.
3. Для каждой публикации достаём DOI.

Версия 3. Версия 2 на живых данных отдавала ноль записей при total_count
в 2,6 млн — сервер игнорировал фильтр по публикациям, когда в запросе стояли
два фильтра __in сразу (document_chembl_id__in и standard_type__in).
Поэтому здесь:
  - одна публикация на запрос, точным совпадением document_chembl_id=...,
    без __in вовсе;
  - тип измерения фильтруется на нашей стороне;
  - запрос отбрасывается, если сервер сообщает подозрительно большое
    total_count: это признак того, что фильтр снова не применился,
    и лучше остановиться, чем скачивать всю базу;
  - режим --probe: один запрос с печатью URL и ответа, чтобы проверить
    работоспособность за три секунды, а не за полтора часа.

Запуск:
    python fetch_chembl.py --probe      # быстрая проверка
    python fetch_chembl.py              # выгрузка, возобновляемая
    python fetch_chembl.py --restart    # начать заново
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_LIMIT = 1000
TIMEOUT = 45
RETRIES = 3
SLEEP = 0.15
MAX_PAGES_PER_DOC = 20
# Если по одной публикации сервер обещает больше записей, чем это,
# значит фильтр не применился. Самые объёмные статьи в ChEMBL — тысячи
# измерений, десятки тысяч означают ошибку запроса.
SANE_TOTAL = 50_000

VIRUS_PATTERNS = ["vaccinia"]
KEEP_TYPES = {"IC50", "EC50", "CC50", "TC50", "TD50", "GI50", "SI"}

ACTIVITY_FIELDS = [
    "activity_id", "molecule_chembl_id", "canonical_smiles",
    "standard_type", "standard_relation", "standard_value", "standard_units",
    "assay_chembl_id", "assay_description", "document_chembl_id",
    "target_chembl_id", "target_organism", "assay_organism", "assay_cell_type",
]
OUT_COLS = ACTIVITY_FIELDS + ["doi", "year", "journal", "title",
                              "is_vaccinia_assay", "source_url"]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def get(session, url, params=None, verbose=False):
    for attempt in range(RETRIES):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            if verbose:
                log(f"      GET {r.url}")
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                log(f"      HTTP {r.status_code}, повтор через {wait}с")
                time.sleep(wait)
                continue
            log(f"      HTTP {r.status_code}: {r.text[:200]}")
            return None
        except requests.RequestException as e:
            wait = 2 ** attempt
            log(f"      {type(e).__name__}, повтор через {wait}с")
            time.sleep(wait)
    return None


# Имена коллекций в ответе ChEMBL не выводятся из имени эндпоинта
# прибавлением «s»: activity → activities, а не activitys. Именно на этом
# версии 1 и 2 читали пустой список и выгружали ноль записей.
COLLECTION_KEY = {
    "activity": "activities",
    "assay": "assays",
    "document": "documents",
    "molecule": "molecules",
    "target": "targets",
}


def extract(data, endpoint):
    """Список записей из ответа ChEMBL, без угадывания имени ключа."""
    stem = endpoint.split(".")[0]
    key = COLLECTION_KEY.get(stem)
    if key and isinstance(data.get(key), list):
        return data[key]
    # запасной путь: первое поле-список, кроме служебных
    for k, v in data.items():
        if k != "page_meta" and isinstance(v, list):
            return v
    return []


def paged(session, endpoint, params, max_pages=None, status=None, verbose=False):
    url = BASE + "/" + endpoint
    first, page = True, 0
    if status is not None:
        status.update(ok=True, truncated=False, insane=False)

    while True:
        data = get(session, url, params if first else None, verbose=verbose and first)
        if data is None:
            if status is not None:
                status["ok"] = False
            return
        meta = data.get("page_meta") or {}

        if first:
            tc = meta.get("total_count")
            if tc is not None and tc > SANE_TOTAL:
                log(f"      total_count = {tc:,} — фильтр не применился, запрос отброшен")
                if status is not None:
                    status.update(ok=False, insane=True, total_count=tc)
                return
            if status is not None:
                status["total_count"] = tc
        first = False

        for it in extract(data, endpoint):
            yield it
        page += 1

        nxt = meta.get("next")
        if not nxt:
            return
        if max_pages and page >= max_pages:
            log(f"      предел {max_pages} страниц, обрезано")
            if status is not None:
                status["truncated"] = True
            return
        url = "https://www.ebi.ac.uk" + nxt
        time.sleep(SLEEP)


# ------------------------------------------------------------------- probe


def probe(session):
    """Быстрая проверка, что запросы вообще работают."""
    log("Проверка 1: поиск ассеев по vaccinia")
    data = get(session, f"{BASE}/assay.json",
               {"assay_organism__icontains": "vaccinia", "limit": 3}, verbose=True)
    if not data:
        log("  НЕ РАБОТАЕТ: ответ не получен")
        return 1
    items = extract(data, "assay.json")
    tc = (data.get("page_meta") or {}).get("total_count")
    log(f"  ключи ответа: {list(data)}")
    log(f"  получено {len(items)}, total_count {tc}")
    if not items:
        log("  НЕ РАБОТАЕТ: список пуст")
        return 1
    doc = items[0].get("document_chembl_id")
    log(f"  пример: {items[0].get('assay_chembl_id')}, публикация {doc}")

    log("\nПроверка 2: активности одной публикации (так работает версия 3)")
    data = get(session, f"{BASE}/activity.json",
               {"document_chembl_id": doc, "limit": 3}, verbose=True)
    if not data:
        log("  НЕ РАБОТАЕТ: ответ не получен")
        return 1
    items = extract(data, "activity.json")
    tc = (data.get("page_meta") or {}).get("total_count")
    log(f"  получено {len(items)}, total_count {tc}")
    if tc and tc > SANE_TOTAL:
        log(f"  ПЛОХО: total_count {tc:,} — фильтр по публикации не применяется")
        return 1
    if items:
        a = items[0]
        log(f"  пример: {a.get('standard_type')} = {a.get('standard_value')} "
            f"{a.get('standard_units')}")
    log("\nВсё в порядке, можно запускать без --probe")
    return 0


# -------------------------------------------------------------------- шаги


def find_assays(session):
    found = {}
    for pat in VIRUS_PATTERNS:
        for field in ("assay_organism__icontains", "description__icontains"):
            log(f"  поиск assay по {field}={pat}")
            n = 0
            for a in paged(session, "assay.json", {field: pat, "limit": PAGE_LIMIT}):
                found[a["assay_chembl_id"]] = a
                n += 1
            log(f"    найдено {n}")
    return found


def fetch_dois(session, doc_ids):
    dois = {}
    for n, d in enumerate(sorted(doc_ids), 1):
        data = get(session, f"{BASE}/document/{d}.json")
        if data:
            dois[d] = {
                "doi": data.get("doi") or "",
                "year": data.get("year") or "",
                "journal": data.get("journal") or "",
                "title": (data.get("title") or "").replace("\n", " "),
            }
        if n % 50 == 0:
            log(f"  DOI: {n}/{len(doc_ids)}")
        time.sleep(SLEEP)
    return dois


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/raw/chembl_activities.csv")
    ap.add_argument("--assays-out", default="../data/raw/chembl_assays.json")
    ap.add_argument("--state", default="../data/raw/.chembl_state.json")
    ap.add_argument("--probe", action="store_true", help="быстрая проверка и выход")
    ap.add_argument("--restart", action="store_true", help="начать заново")
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "vacv-si-hackathon/1.0 (research use)"

    if args.probe:
        return probe(session)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state)

    state = {"done_docs": [], "failed_docs": []}
    if state_path.exists() and not args.restart:
        state = json.loads(state_path.read_text())
        log(f"Возобновление: обработано публикаций {len(state['done_docs'])}")
    elif args.restart:
        if out.exists():
            out.unlink()
        if state_path.exists():
            state_path.unlink()
        state = {"done_docs": [], "failed_docs": []}

    # ------------------------------------------------------------------ шаг 1
    assays_path = Path(args.assays_out)
    if assays_path.exists() and not args.restart:
        assays = {a["assay_chembl_id"]: a for a in json.loads(assays_path.read_text())}
        log(f"Шаг 1: ассеи из кэша, {len(assays)}")
    else:
        log("Шаг 1: поиск ассеев по vaccinia")
        assays = find_assays(session)
        log(f"  уникальных ассеев: {len(assays)}")
        if not assays:
            log("Ничего не найдено. Запусти с --probe, чтобы понять причину.")
            return 1
        assays_path.parent.mkdir(parents=True, exist_ok=True)
        assays_path.write_text(json.dumps(list(assays.values()), ensure_ascii=False))

    vaccinia_assays = set(assays)
    all_docs = sorted({a["document_chembl_id"] for a in assays.values()
                       if a.get("document_chembl_id")})
    done = set(state["done_docs"])
    todo = [d for d in all_docs if d not in done]
    log(f"Шаг 2: публикаций {len(all_docs)}, осталось {len(todo)}")

    # ------------------------------------------------------------------ шаг 2
    write_header = not out.exists()
    f = out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    t0, kept_total, insane = time.time(), 0, 0
    try:
        for i, doc in enumerate(todo, 1):
            status = {}
            rows = list(paged(session, "activity.json",
                              {"document_chembl_id": doc, "limit": PAGE_LIMIT},
                              max_pages=MAX_PAGES_PER_DOC, status=status,
                              verbose=(i == 1)))
            kept = [a for a in rows if a.get("standard_type") in KEEP_TYPES]
            for a in kept:
                row = {k: a.get(k, "") for k in ACTIVITY_FIELDS}
                row["is_vaccinia_assay"] = int(a.get("assay_chembl_id") in vaccinia_assays)
                row["source_url"] = f"{BASE}/activity/{a.get('activity_id')}"
                writer.writerow(row)
            f.flush()
            kept_total += len(kept)

            if status.get("insane"):
                insane += 1
                if insane >= 3:
                    log("\nТри запроса подряд вернули всю базу вместо выборки. "
                        "Остановка. Запусти с --probe и пришли вывод.")
                    break
            else:
                insane = 0

            if status.get("ok"):
                done.add(doc)
            else:
                state["failed_docs"].append(doc)

            if i % 10 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1)
                eta = (len(todo) - i) / rate / 60
                log(f"  [{i}/{len(todo)}] {doc}: всего отобрано {kept_total}"
                    f"   осталось примерно {eta:.0f} мин")
                state["done_docs"] = sorted(done)
                state_path.write_text(json.dumps(state))
            time.sleep(SLEEP)
    except KeyboardInterrupt:
        log("\nПрервано. Прогресс сохранён, запусти ту же команду для продолжения.")
        return 130
    finally:
        state["done_docs"] = sorted(done)
        state_path.write_text(json.dumps(state))
        f.close()

    if kept_total == 0:
        log("\nНе отобрано ни одной записи. Запусти с --probe и пришли вывод.")
        return 1

    # ------------------------------------------------------------------ шаг 3
    log(f"Шаг 3: DOI для {len(all_docs)} публикаций")
    dois = fetch_dois(session, all_docs)
    log(f"  получено DOI: {sum(1 for v in dois.values() if v['doi'])}")

    import pandas as pd
    df = pd.read_csv(out)
    meta = pd.DataFrame.from_dict(dois, orient="index")
    df = df.drop(columns=[c for c in ("doi", "year", "journal", "title") if c in df])
    df = df.join(meta, on="document_chembl_id")
    df.to_csv(out, index=False)

    log(f"\nГотово: {len(df)} строк в {out}")
    uniq = df["canonical_smiles"].nunique() if "canonical_smiles" in df else 0
    log(f"  уникальных молекул: {uniq}")
    if state["failed_docs"]:
        log(f"  не выгружено публикаций: {len(set(state['failed_docs']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
