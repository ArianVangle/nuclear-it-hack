"""
Scaffold-разбиение по каркасам Мурко.

Зачем не случайное разбиение. В выборке много серий аналогов — соединений
с общим каркасом и разными заместителями. При случайном split аналоги
из одной серии попадают и в обучение, и в валидацию, модель запоминает серию,
и локальная оценка оказывается заметно оптимистичнее лидерборда.
Scaffold-разбиение держит всю серию в одном фолде и измеряет то, что нас
действительно интересует: предсказание для структурно нового соединения.
"""

from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def murcko_scaffold(smiles):
    """
    Каркас Мурко. Для ациклических молекул каркас пустой — такие соединения
    получают собственную группу, иначе все они слиплись бы в одну огромную.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return smiles
    return scaf if scaf else smiles


def _check_smiles_list(smiles_list):
    """
    Защита от ошибки, которая стоила нам дня работы.

    Если сюда передать DataFrame, Python переберёт его ПО ИМЕНАМ КОЛОНОК,
    а не по строкам: функция увидит десяток «молекул» с названиями
    'smiles', 'Title', 'strain' и разложит их по фолдам. При этом
    len(DataFrame) вернёт число строк, маска получится верного размера,
    ничего не упадёт — и валидироваться будет 10 строк из 320,
    а метрика выродится в оценку константного нуля.

    Правильный вызов: scaffold_kfold(df["smiles"].tolist(), ...)
    """
    if hasattr(smiles_list, "columns"):
        raise TypeError(
            "передан DataFrame, а нужен список SMILES: "
            "вызывайте scaffold_kfold(df['smiles'].tolist(), ...)"
        )
    bad = [s for s in list(smiles_list)[:50] if not isinstance(s, str)]
    if bad:
        raise TypeError("ожидались строки SMILES, получено например %r" % (bad[0],))
    return list(smiles_list)


def scaffold_groups(smiles_list):
    """smiles -> индексы молекул с этим каркасом."""
    smiles_list = _check_smiles_list(smiles_list)
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        groups[murcko_scaffold(smi) or f"__bad_{i}"].append(i)
    return dict(groups)


def scaffold_kfold(smiles_list, n_splits=5, seed=0):
    """
    Разбиение на фолды по каркасам. Группы перемешиваются, затем жадно
    раскладываются в наименее заполненный фолд — так размеры фолдов
    получаются сопоставимыми, несмотря на очень разный размер групп.

    Возвращает список пар (train_idx, val_idx).
    """
    groups = scaffold_groups(smiles_list)
    keys = list(groups)

    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    # Крупные группы раскладываем первыми, иначе последняя большая группа
    # перекосит фолд, в который попадёт.
    keys.sort(key=lambda k: len(groups[k]), reverse=True)

    folds = [[] for _ in range(n_splits)]
    for k in keys:
        target = min(range(n_splits), key=lambda f: len(folds[f]))
        folds[target].extend(groups[k])

    n = len(smiles_list)
    out = []
    for f in range(n_splits):
        val = np.array(sorted(folds[f]), dtype=int)
        mask = np.ones(n, dtype=bool)
        mask[val] = False
        out.append((np.where(mask)[0], val))
    return out


def repeated_scaffold_kfold(smiles_list, n_splits=5, n_repeats=5, base_seed=0):
    """Повторы разбиения с разными сидами. Возвращает (seed, folds)."""
    for r in range(n_repeats):
        seed = base_seed + r
        yield seed, scaffold_kfold(smiles_list, n_splits=n_splits, seed=seed)


def describe_split(smiles_list, n_splits=5, seed=0):
    """Диагностика: насколько разбиение сбалансировано и изолированы ли каркасы."""
    groups = scaffold_groups(smiles_list)
    folds = scaffold_kfold(smiles_list, n_splits, seed)
    sizes = [len(v) for _, v in folds]
    big = sum(1 for g in groups.values() if len(g) > 1)

    # Каркас не должен встречаться одновременно в train и val
    scaf = np.array([murcko_scaffold(s) for s in smiles_list], dtype=object)
    leaks = 0
    for tr, va in folds:
        leaks += len(set(scaf[tr]) & set(scaf[va]))

    return {
        "n_molecules": len(smiles_list),
        "n_scaffolds": len(groups),
        "n_scaffolds_multi": big,
        "largest_scaffold": max(len(g) for g in groups.values()),
        "fold_sizes": sizes,
        "scaffold_leaks": leaks,
    }


def random_kfold(smiles_list, n_splits=5, seed=0):
    """
    Обычное случайное разбиение.

    Нужно как ВТОРАЯ оценка рядом со scaffold-split, а не вместо него.
    Причина: у 66,8% строк настоящего test каркас присутствует в train,
    то есть тест не является scaffold-split. Scaffold-CV моделирует более
    трудную задачу и даёт систематически пессимистичную оценку
    (15,6 млн против 10,1 млн на публичном лидерборде).

    Scaffold-CV остаётся мерой устойчивости к структурно новым соединениям,
    случайное разбиение — калибровкой под лидерборд. Решения принимаются
    по обоим: улучшение, которое видно только в случайном split,
    скорее всего, запоминание серии аналогов.
    """
    # Эта функция использует только длину, поэтому DataFrame она пережила бы
    # молча и правильно. Проверка всё равно стоит: иначе одни и те же вызовы
    # корректны для random и сломаны для scaffold, что и случилось.
    smiles_list = _check_smiles_list(smiles_list)
    n = len(smiles_list)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    out = []
    for f in range(n_splits):
        val = np.sort(idx[f::n_splits])
        mask = np.ones(n, dtype=bool)
        mask[val] = False
        out.append((np.where(mask)[0], val))
    return out


def repeated_random_kfold(smiles_list, n_splits=5, n_repeats=5, base_seed=0):
    for r in range(n_repeats):
        yield base_seed + r, random_kfold(smiles_list, n_splits, base_seed + r)
