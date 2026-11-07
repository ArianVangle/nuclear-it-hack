"""
Признаки из SMILES.

Только структура молекулы. Колонки CC50_uM и IC50_uM_VACV запрещены
правилами соревнования и сюда не попадают — за этим следит check_no_forbidden.

Поля strain и Culture намеренно не используются: формально они не запрещены,
но это признаки протокола эксперимента, а не молекулы, и на защите их могут
счесть обходом условия. Вопрос задан организаторам; если разрешат, добавить
их сюда несложно.
"""

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

RDLogger.DisableLog("rdApp.*")

MORGAN_BITS = 2048
MORGAN_RADIUS = 2

_morgan = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)

# Дескрипторы, которые считаются быстро и не дают NaN на типичных молекулах
_DESC = [
    "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
    "HeavyAtomCount", "NHOHCount", "NOCount", "NumSaturatedRings",
    "NumAliphaticRings", "LabuteASA", "BalabanJ", "BertzCT",
    "MolMR", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v",
    "MaxPartialCharge", "MinPartialCharge",
    "NumHeteroatoms", "NumValenceElectrons", "qed",
]
_DESC_FN = [(n, getattr(Descriptors, n)) for n in _DESC if hasattr(Descriptors, n)]

# Полный набор дескрипторов RDKit (217 штук против 32 отобранных вручную).
#
# Зачем он понадобился. Дескрипторы оказались первым блоком признаков,
# который реально помог поверх отпечатков: устойчивая метрика упала на 13,5%
# в классификаторе хвоста и на 16% в основной модели. Механизм в том, что
# селективность есть сила, делённая на токсичность, токсичность тянется
# за липофильностью и размером, а битами подструктур «logP в таком-то
# диапазоне» не выразить.
#
# В отобранных 32 нет как раз самого подходящего: семейств PEOE_VSA
# и SlogP_VSA (распределение частичного заряда и липофильности по площади
# поверхности), Chi/Kappa (топологическая форма) и EState. Это прямые
# характеристики взаимодействия с мембраной, то есть цитотоксичности.
#
# Риск обратный: 217 признаков на 320 обучающих молекул. Поэтому блок
# добавлен отдельно, а не вместо `desc`, и принимается только по замеру.
_DESC_ALL_FN = list(Descriptors._descList)


def _mol(smi):
    return Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi.strip() else None


def morgan(smiles_list):
    """ECFP4, бинарный вектор 2048."""
    X = np.zeros((len(smiles_list), MORGAN_BITS), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = _mol(smi)
        if mol is None:
            continue
        fp = _morgan.GetFingerprint(mol)
        arr = np.zeros((MORGAN_BITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X, [f"morgan_{i}" for i in range(MORGAN_BITS)]


def maccs(smiles_list):
    """MACCS keys, 167 бит."""
    X = np.zeros((len(smiles_list), 167), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = _mol(smi)
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X, [f"maccs_{i}" for i in range(167)]


def descriptors(smiles_list):
    """Физико-химические дескрипторы RDKit."""
    X = np.full((len(smiles_list), len(_DESC_FN)), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        mol = _mol(smi)
        if mol is None:
            continue
        for j, (_, fn) in enumerate(_DESC_FN):
            try:
                v = fn(mol)
                X[i, j] = v if np.isfinite(v) else np.nan
            except Exception:
                pass
    # LightGBM умеет работать с NaN, но бесконечности нужно убрать
    X[~np.isfinite(X)] = np.nan
    return X, [f"desc_{n}" for n, _ in _DESC_FN]


def descriptors_all(smiles_list):
    """Все дескрипторы RDKit. См. комментарий к _DESC_ALL_FN."""
    X = np.full((len(smiles_list), len(_DESC_ALL_FN)), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        mol = _mol(smi)
        if mol is None:
            continue
        for j, (_, fn) in enumerate(_DESC_ALL_FN):
            try:
                v = fn(mol)
                X[i, j] = v if np.isfinite(v) else np.nan
            except Exception:
                pass
    X[~np.isfinite(X)] = np.nan
    return X, [f"dall_{n}" for n, _ in _DESC_ALL_FN]


def _bits(gen, smiles_list, n, prefix, counts=False):
    X = np.zeros((len(smiles_list), n), dtype=np.int32 if counts else np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = _mol(smi)
        if mol is None:
            continue
        fp = gen.GetCountFingerprint(mol) if counts else gen.GetFingerprint(mol)
        arr = np.zeros((n,), dtype=np.int32 if counts else np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X, [f"{prefix}_{i}" for i in range(n)]


# Отпечатки других семейств. Morgan кодирует круговое окружение атома;
# эти кодируют другое, поэтому могут нести то, чего в Morgan нет:
#   counts   — тот же Morgan, но с кратностью фрагмента, а не фактом наличия
#   fcfp     — Morgan по фармакофорным ролям атомов (донор, акцептор,
#              ароматика), а не по элементам: обобщает между похожими
#              по функции, но разными по составу фрагментами
#   atompair — пары атомов с топологическим расстоянием между ними
#   torsion  — четвёрки связанных атомов
#   rdkfp    — пути по графу молекулы
def morgan_chiral(smiles_list):
    """
    Отпечаток Моргана С УЧЁТОМ стереохимии.

    Зачем понадобился. Все остальные наши отпечатки строятся без
    хиральности, поэтому два эпимера для модели — одна и та же молекула.
    Это дало дорогую ошибку: тестовой строке 233 модель поставила 13 053,
    потому что её плоская структура совпадает с обучающей молекулой,
    у которой SI = 36 067. Между тем на самих обучающих данных видно,
    что внутри одной серии значения расходятся до 183 раз, то есть
    стереохимия для активности решает.

    Блок добавлен отдельно, а не заменой: принимается только по замеру.
    """
    gen = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS,
                             includeChirality=True)
    return _bits(gen, smiles_list, MORGAN_BITS, "mgchi")


def morgan_counts_chiral(smiles_list):
    """Счётный Морган с учётом стереохимии. См. morgan_chiral."""
    return _bits(GetMorganGenerator(radius=2, fpSize=MORGAN_BITS,
                                    includeChirality=True),
                 smiles_list, MORGAN_BITS, "mgcchi", counts=True)


def torsion_chiral(smiles_list):
    """Торсионный отпечаток с учётом стереохимии. См. morgan_chiral."""
    from rdkit.Chem.rdFingerprintGenerator import GetTopologicalTorsionGenerator
    return _bits(GetTopologicalTorsionGenerator(fpSize=MORGAN_BITS,
                                                includeChirality=True),
                 smiles_list, MORGAN_BITS, "torchi")


def morgan_counts(smiles_list):
    return _bits(GetMorganGenerator(radius=2, fpSize=MORGAN_BITS), smiles_list,
                 MORGAN_BITS, "morganc", counts=True)


def fcfp(smiles_list):
    from rdkit.Chem.rdFingerprintGenerator import GetMorganFeatureAtomInvGen
    gen = GetMorganGenerator(radius=2, fpSize=MORGAN_BITS,
                             atomInvariantsGenerator=GetMorganFeatureAtomInvGen())
    return _bits(gen, smiles_list, MORGAN_BITS, "fcfp")


def atompair(smiles_list):
    from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator
    return _bits(GetAtomPairGenerator(fpSize=2048), smiles_list, 2048, "ap")


def torsion(smiles_list):
    from rdkit.Chem.rdFingerprintGenerator import GetTopologicalTorsionGenerator
    return _bits(GetTopologicalTorsionGenerator(fpSize=2048), smiles_list, 2048, "tt")


def rdkfp(smiles_list):
    from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator
    return _bits(GetRDKitFPGenerator(fpSize=2048), smiles_list, 2048, "rdk")


BUILDERS = {"morgan": morgan, "maccs": maccs, "desc": descriptors,
            "descall": descriptors_all,
            "morganchi": morgan_chiral,
            "morgancchi": morgan_counts_chiral,
            "torsionchi": torsion_chiral,
            "morganc": morgan_counts, "fcfp": fcfp, "atompair": atompair,
            "torsion": torsion, "rdkfp": rdkfp}


def build(smiles_list, blocks=("morgan", "desc")):
    """
    Собирает матрицу признаков из выбранных блоков.
    Возвращает (X, имена_признаков).
    """
    from data import check_no_forbidden

    parts, names = [], []
    for b in blocks:
        if b not in BUILDERS:
            raise ValueError(f"неизвестный блок признаков: {b}")
        Xb, nb = BUILDERS[b](smiles_list)
        parts.append(Xb.astype(np.float64))
        names.extend(nb)
    X = np.hstack(parts) if parts else np.zeros((len(smiles_list), 0))
    check_no_forbidden(names)
    return X, names


def drop_constant(X_train, X_other_list, names):
    """
    Убирает признаки, постоянные на обучающей части. На 2048 битах Morgan
    таких обычно больше половины: они только замедляют обучение.
    Порог считается ТОЛЬКО по train, чтобы не подсматривать в валидацию.
    """
    keep = X_train.std(axis=0) > 0
    out = [X_train[:, keep]] + [X[:, keep] for X in X_other_list]
    return out, [n for n, k in zip(names, keep) if k]
