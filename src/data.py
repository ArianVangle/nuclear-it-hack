"""
Загрузка и подготовка данных соревнования.

Главное решение, зашитое здесь: SI = 10000 — это не измерение, а заглушка.
В train таких строк 81 из 320, и во ВСЕХ из них отсутствует IC50. Это значит
«ингибирование не обнаружено», то есть вещество неактивно. Мы выделяем их
в отдельный класс, потому что метрика MSE делает точность определения этого
класса важнее качества самой регрессии.

Колонки CC50_uM и IC50_uM_VACV запрещены правилами соревнования и здесь
не используются ни как признаки, ни косвенно.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ВНИМАНИЕ: организаторы сменили кодировку (train_v2 от 07.11.2026).
# «Активность не обнаружена» раньше записывалось как SI = 10000,
# теперь как SI = 0. Это меняет задачу радикально: раньше такие строки
# давали основную часть ошибки, теперь они почти бесплатны, а 99%
# дисперсии цели приходится на 7 молекул с SI > 1000.
SENTINEL = 0.0
TARGET = "SI VACV"
# Колонки, которые нельзя использовать для построения модели
FORBIDDEN = ["CC50_uM", "IC50_uM_VACV"]


def canonical_smiles(smi):
    """Канонический SMILES. None, если молекула не читается."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def load_competition(data_dir="data", verbose=True):
    """Читает train.csv и test.csv, канонизирует SMILES, размечает заглушки."""
    data_dir = Path(data_dir)
    train_file = "train_v2.csv" if (data_dir / "train_v2.csv").exists() else "train.csv"
    train = pd.read_csv(data_dir / train_file)
    test = pd.read_csv(data_dir / "test.csv")

    for df in (train, test):
        df["smiles"] = df["SMILES"].map(canonical_smiles)

    n_bad_train = train["smiles"].isna().sum()
    n_bad_test = test["smiles"].isna().sum()
    if n_bad_train:
        train = train[train["smiles"].notna()].copy()
    if n_bad_test:
        # В test строки терять нельзя — нужен ответ на каждый index.
        # Оставляем как есть, модель получит для них запасное предсказание.
        pass

    train["y"] = train[TARGET].astype(float)
    train["is_sentinel"] = (train["y"] == SENTINEL).astype(int)

    if verbose:
        print(f"train: {len(train)} строк ({train_file}), test: {len(test)} строк")
        if n_bad_train or n_bad_test:
            print(f"  нечитаемых SMILES: train {n_bad_train}, test {n_bad_test}")
        n_s = int(train["is_sentinel"].sum())
        print(f"  нулевых SI: {n_s} ({n_s / len(train):.1%})")
        act = train.loc[train["is_sentinel"] == 0, "y"]
        print(f"  ненулевых: {len(act)}, медиана SI {act.median():.2f}, максимум {act.max():.1f}")
        hi = act > 1000
        print(f"  из них SI>1000: {int(hi.sum())} — на них приходится почти вся дисперсия")
        dup = train["smiles"].duplicated().sum()
        overlap = test["smiles"].isin(set(train["smiles"])).sum()
        print(f"  дубликатов в train: {dup}, пересечение test/train: {overlap}")

        # Проверка: нулевые SI — это строки без IC50.
        # Сами колонки в модели не используются, это только диагностика.
        if "IC50_uM_VACV" in train:
            no_ic50 = train["IC50_uM_VACV"].isna()
            agree = (no_ic50 == train["is_sentinel"].astype(bool)).mean()
            print(f"  диагностика: совпадение «нет IC50» и «SI=0»: {agree:.1%}")

    # Структурная защита: из ТЕСТА запрещённые колонки удаляются физически.
    #
    # Раньше они просто лежали в таблице рядом с остальными и не
    # использовались по договорённости. Этого мало: в test.csv CC50 заполнена
    # у 595 строк из 597, IC50 у 470, а целевая переменная равна их отношению
    # (проверено на train: совпадение лучше 1% у всех 237 строк, где заполнены
    # обе). То есть одна невнимательная строка кода — и модель получает ответ,
    # а мы этого даже не заметим.
    #
    # В TRAIN они остаются: там это легальные ЦЕЛИ вспомогательных моделей
    # (организаторы подтвердили, что предсказывать CC50 и IC50 и делить одно
    # на другое можно). Запрещено использовать их как признаки и брать
    # из теста — и то и другое теперь невозможно физически.
    test = test.drop(columns=[c for c in FORBIDDEN if c in test.columns])
    if verbose:
        print(f"  из теста удалены запрещённые колонки: {FORBIDDEN}")

    return train, test


def load_external(path="data/external.csv", exclude_test=True, verbose=True):
    """
    Подключает внешний датасет, если он собран. Молекулы, пересекающиеся
    с test, по умолчанию выбрасываются — иначе локальная оценка будет завышена.
    """
    path = Path(path)
    if not path.exists():
        if verbose:
            print(f"внешних данных нет ({path})")
        return None

    ext = pd.read_csv(path)
    n0 = len(ext)
    ext = ext[ext["SI"].notna()].copy()
    ext["smiles"] = ext["smiles"].map(canonical_smiles)
    ext = ext[ext["smiles"].notna()].copy()

    if exclude_test and "in_test" in ext:
        n_leak = int(ext["in_test"].sum())
        ext = ext[ext["in_test"] == 0].copy()
        if verbose and n_leak:
            print(f"  исключено молекул, пересекающихся с test: {n_leak}")

    ext["y"] = ext["SI"].astype(float)
    if "is_sentinel" not in ext:
        ext["is_sentinel"] = (ext["y"] == SENTINEL).astype(int)

    if verbose:
        print(f"внешних данных: {len(ext)} молекул (из {n0} строк файла)")
        n_s = int(ext["is_sentinel"].sum())
        print(f"  заглушек: {n_s} ({n_s / max(len(ext), 1):.1%})")

    return ext


def check_no_forbidden(feature_names):
    """Страховка: ни один признак не должен быть построен на CC50 или IC50."""
    bad = [f for f in feature_names if any(k.lower() in str(f).lower()
                                           for k in ("cc50", "ic50"))]
    if bad:
        raise ValueError(
            f"Запрещённые правилами соревнования признаки: {bad}. "
            f"Колонки {FORBIDDEN} использовать нельзя."
        )


if __name__ == "__main__":
    tr, te = load_competition()
    load_external()
