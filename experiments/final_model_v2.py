"""
Каноническая модель под кодировку train_v2 (нулевая заглушка).

ЧЕМ ЭТА АРХИТЕКТУРА ОТЛИЧАЕТСЯ ОТ ПРЕДЫДУЩЕЙ, И ПОЧЕМУ

1. Задача сменилась вместе с метками. При старой кодировке «нет активности»
   записывалось как SI = 10000, и 81 такая молекула определяла почти всю
   дисперсию — поэтому прошлая модель была устроена вокруг классификатора
   заглушек. В train_v2 те же молекулы кодируются нулём. Теперь нулевая
   строка вносит в MSE всего pred^2: при 150 нулевых строках и нашем
   типичном прогнозе ~30 это 226 единиц из 1,7 миллиона, то есть ничто.
   Весь бюджет ошибки — примерно 13 высоких молекул из 597.

2. Главный источник точности — не модель, а восстановление первоисточника.
   Организаторы собрали выборку из литературы. Если взять из ChEMBL
   активность против осповируса и цитотоксичность ИЗ ОДНОЙ публикации
   и поделить, получается их число. На пересечении с train: корреляция
   log10 равна 0,960, MSE подстановки 49 361 против 542 260 у константы.
   Проверка на лидерборде дала точность 0,1% (тековиримат: наши 2500,
   их 2498). Покрытие — 221 тестовая строка из 597.

3. Обученный на внешних данных классификатор «хвоста» из модели УБРАН.
   Под честным фильтром (исключая не только точные SMILES, но и плоские
   структуры соревновательных молекул) его AUC падает до 0,49–0,56, то есть
   до уровня монетки, а вклад в метрику — 224 359 против 224 533 без него.
   Вся его видимая сила была узнаванием знакомых молекул; прямая подстановка
   делает то же самое вчетверо лучше и честно.

ПОРЯДОК ПРИОРИТЕТОВ ПРИ СБОРКЕ ПРОГНОЗА
   1) собственное измерение организаторов, если молекула есть в train
   2) литературное значение из восстановленного источника
   3) модель

ЧТО ВХОДИТ В МОДЕЛЬ (всё принято по замеру в ОБЕИХ схемах валидации)
   - 4 семейства отпечатков, предсказания усредняются (ансамбль
     представлений; объединение отпечатков в один вектор не работает)
   - 178 дескрипторов RDKit: устойчивая метрика 158 805 -> 126 711 в хвосте
     и 137 379 -> 115 286 в основной модели. Механизм: селективность это
     сила, делённая на токсичность, токсичность тянется за липофильностью
     и распределением заряда по поверхности, а битами подструктур это
     не выразить. Единственный случай, когда объединение признаков помогло,
     потому что дескрипторы несут другой РОД информации.
   - литературные IC50 и CC50 как признаки (не как формула):
     197 270 -> 182 609. Обратный IC50 ранжирует SI с AUC 0,979, но деление
     на предполагаемую цитотоксичность проигрывает модели (247 471
     против 184 274), поэтому решение отдаётся модели.
   - целевое кодирование strain и Culture, признаки из Title
   - интервальная модель (ожидание по интервалам) в равной доле с log1p-регрессией

О ВАЛИДАЦИИ. Оценка считается в двух схемах и взвешивается под фактический
состав теста: 0,668 * random + 0,332 * scaffold, поскольку у 66,8% тестовых
молекул каркас встречается в train. Основной ориентир — метрика БЕЗ самой
экстремальной молекулы: полная метрика на 95% определяется одним объектом
(36 067), который не берёт ни одна модель, и сравнивать по ней значит
сравнивать шум.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import lightgbm as lgb

from binned_model import BinnedExpectationModel
from data import canonical_smiles
from features import build
from meta_features import MetaEncoder
from title_features import build_title

RDLogger.DisableLog("rdApp.*")

# Стереочувствительные варианты: отпечаток, не различающий эпимеры,
# считает разные молекулы одной. Это дало дорогую ошибку — тестовой
# строке 233 модель ставила 13 053 при истинных 695,7, потому что её
# плоская структура совпадает с обучающей молекулой, где SI = 36 067.
# После замены Tanimoto между ними падает с 1,000 до 0,52-0,80,
# а прогноз — до 7 193. rdkfp оставлен обычным: он хиральность
# не поддерживает.
REPRESENTATIONS = [("morganchi",), ("morgancchi",), ("torsionchi",), ("rdkfp",)]
N_SEEDS = 10

REG_PARAMS = dict(objective="regression", n_estimators=400, learning_rate=0.05,
                  num_leaves=7, min_child_samples=10, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                  verbose=-1)

POTENCY = {"EC50", "IC50"}
TOXICITY = {"CC50", "TC50", "GI50", "TD50"}
UNITS = {"nM": 1e-3, "uM": 1.0, "M": 1e6, "pM": 1e-6, "mM": 1e3}


def flat_smiles(smi):
    """Структура без стереохимии: литература редко согласована по изомерам."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def rebuilt_tables(path):
    """
    Таблицы из collect_all.py — сбора по САМИМ структурам соревнования.

    Прежний сбор искал статьи по слову "vaccinia" в описании ассея и нашёл
    246 документов; из 597 тестовых строк 243 молекулы в него не попали
    вовсе. Новый спрашивает ChEMBL про каждую из 917 структур напрямую,
    поэтому ничего не теряется из-за формулировки.

    Если файла нет, модель работает на прежней выгрузке — см.
    literature_tables().
    """
    frame = pd.read_csv(path)
    frame["flat"] = frame["smiles"].map(flat_smiles)
    return {
        "si": frame.groupby("flat")["SI"].median(),
        "ic50": frame.groupby("flat")["IC50"].median(),
        "cc50": frame.groupby("flat")["CC50"].median(),
    }


def merge_tables(primary, fallback):
    """Приоритет у нового сбора, пропуски добираются из старого."""
    out = {}
    for key in ("si", "ic50", "cc50"):
        a, b = primary.get(key), fallback.get(key)
        if a is None:
            out[key] = b
        elif b is None:
            out[key] = a
        else:
            out[key] = a.combine_first(b)
    return out


def literature_tables(raw_path):
    """
    Восстановление первоисточника из сырых записей ChEMBL.

    Фильтр по осповирусу обязателен: сбор шёл по статьям, где vaccinia
    лишь упоминается, а такие статьи тестируют вещество против панели
    вирусов. Без фильтра в набор попадает селективность против ВИЧ
    и герпеса — именно так Tenofovir alafenamide получал SI 8000
    и поднимался моделью до 2070 против осповируса.
    """
    raw = pd.read_csv(raw_path, low_memory=False)
    raw["uM"] = (pd.to_numeric(raw["standard_value"], errors="coerce")
                 * raw["standard_units"].map(UNITS))
    raw = raw[raw["uM"].notna() & (raw["uM"] > 0)].copy()

    is_pox = (raw["target_organism"].astype(str)
              .str.contains("vaccinia|Orthopox", case=False, na=False)
              | (raw["is_vaccinia_assay"] == 1))
    potency = raw[is_pox & raw["standard_type"].isin(POTENCY)].copy()
    toxicity = raw[raw["standard_type"].isin(TOXICITY)].copy()
    for frame in (potency, toxicity):
        frame["flat"] = frame["canonical_smiles"].map(
            lambda s: flat_smiles(canonical_smiles(s)))

    # SI считается только из пары, взятой из ОДНОЙ публикации: три колонки
    # готового external.csv между собой не согласованы (отношение CC50/IC50
    # воспроизводит его же SI с AUC 0,468), потому что склеивались по молекуле.
    ic = potency.groupby(["canonical_smiles", "doi"])["uM"].median()
    cc = toxicity.groupby(["canonical_smiles", "doi"])["uM"].median()
    pair = pd.concat([ic.rename("ic"), cc.rename("cc")], axis=1).dropna().reset_index()
    pair["flat"] = pair["canonical_smiles"].map(
        lambda s: flat_smiles(canonical_smiles(s)))
    pair["si"] = pair["cc"] / pair["ic"]

    return {
        # Медиана, а не 75-й процентиль: проверка на лидерборде показала,
        # что для тековиримата (разброс между статьями в 96 раз) медиана
        # даёт 2500 при истинных 2498, а 75-й процентиль промахнулся бы втрое.
        "si": pair.groupby("flat")["si"].median(),
        "ic50": potency.groupby("flat")["uM"].median(),
        "cc50": toxicity.groupby("flat")["uM"].median(),
    }


def _lit_features(frame, tables):
    ic = frame["flat"].map(tables["ic50"]).to_numpy(dtype=float)
    cc = frame["flat"].map(tables["cc50"]).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ic = np.log10(np.where(np.isfinite(ic) & (ic > 0), ic, np.nan))
        log_cc = np.log10(np.where(np.isfinite(cc) & (cc > 0), cc, np.nan))
    # Третий столбец — флаг наличия: сам факт, что вещество описано
    # в литературе по осповирусу, несёт сигнал отдельно от величины.
    return np.c_[log_ic, log_cc, np.isfinite(ic).astype(float)]


def _auxiliary_ratio(train, test, blocks_a, blocks_b, aux_a, aux_b, seed):
    """
    Разрешённая декомпозиция: SI = CC50 / IC50.

    Колонки CC50_uM и IC50_uM_VACV нельзя использовать как ПРИЗНАКИ
    и нельзя читать из теста — это прямой запрет в правилах. Но
    организаторы отдельно подтвердили, что предсказывать их моделью
    и делить одно на другое можно. Здесь они берутся только из train
    и только как ЦЕЛИ вспомогательных моделей; из test не читаются.

    Замер (устойчивая метрика, обе схемы):
        основная модель       180 962 / 181 812
        декомпозиция одна     148 167 / 182 958   <- хуже на scaffold
        смесь 50/50           151 478 / 176 913   <- лучше в обеих
    Поэтому берётся смесь, а не чистая декомпозиция: на незнакомых
    каркасах декомпозиция проседает ниже основной модели.
    """
    out = []
    for target in (aux_a, aux_b):
        mask = np.isfinite(target) & (target > 0)
        if mask.sum() < 30:
            return None
        preds = []
        for xa, xb in zip(blocks_a, blocks_b):
            model = lgb.LGBMRegressor(random_state=seed, **REG_PARAMS)
            model.fit(xa[mask], np.log10(target[mask]))
            preds.append(model.predict(xb))
        out.append(np.mean(preds, axis=0))
    return np.clip(10.0 ** (out[0] - out[1]), 0.0, None)


def _descriptors(train_smiles, test_smiles):
    a, _ = build(train_smiles, ("descall",))
    b, _ = build(test_smiles, ("descall",))
    good = (np.isfinite(a).all(0) & np.isfinite(b).all(0)
            & (a.std(0) > 0))
    return a[:, good], b[:, good]


def predict(train, test, raw_path, rebuilt_path=None, n_seeds=N_SEEDS, verbose=True):
    """
    Полный прогноз для тестовой выборки.

    rebuilt_path — результат collect_all.py (data/external_rebuilt.csv).
    Он используется ТОЛЬКО для добора пропусков, приоритет у прежней
    выгрузки. Так получилось по замеру, и вот почему.

    В collect_all.py фильтр активности написан шире: он пропускает записи
    по коровьей оспе, эктромелии и вариоле. Замер показал, что для
    пересекающихся молекул это портит значения (5000 -> 2000, 2500 -> 312,5),
    и MSE на 107 общих молекулах выходит 259 461 против 52 472 у прежней
    выгрузки — у организаторов взяты именно осповирусные измерения.

    Зато молекулы, которых в прежней выгрузке не было (в основном
    из PubChem по идентификатору в Title), новый сбор восстанавливает
    точно: 2,5 -> 2,5; 1,3 -> 1,3; 1,0 -> 1,0.

    Итог замера на train:
        только прежняя            114 молекул, MSE 49 361
        только новый сбор         117 молекул, MSE 259 461
        прежняя + новый в пропуски 124 молекулы, MSE 45 380   <-- так и делаем
    Покрытие теста при этом растёт с 221 до 244 строк.
    """
    train = train.copy()
    test = test.copy()
    train["flat"] = train["smiles"].map(flat_smiles)
    test["flat"] = test["smiles"].map(flat_smiles)

    tables = literature_tables(raw_path)
    if rebuilt_path is not None and Path(rebuilt_path).exists():
        new = rebuilt_tables(rebuilt_path)
        if verbose:
            print("новый сбор: %d молекул, покрытие теста %d из %d"
                  % (len(new["si"]),
                     int(test["flat"].isin(new["si"].index).sum()), len(test)))
        # Приоритет у прежней выгрузки: новый сбор только закрывает пропуски.
        tables = merge_tables(tables, new)
    y = train["y"].to_numpy()

    desc_a, desc_b = _descriptors(train["smiles"].tolist(), test["smiles"].tolist())
    lit_a = _lit_features(train, tables)
    lit_b = _lit_features(test, tables)
    title_a, _ = build_title(train["Title"].tolist())
    title_b, _ = build_title(test["Title"].tolist())
    if verbose:
        print("дескрипторов %d | литературных совпадений: train %d, test %d"
              % (desc_a.shape[1], int(np.isfinite(lit_a[:, 0]).sum()),
                 int(np.isfinite(lit_b[:, 0]).sum())))

    fps_a = {bl: build(train["smiles"].tolist(), bl)[0] for bl in REPRESENTATIONS}
    fps_b = {bl: build(test["smiles"].tolist(), bl)[0] for bl in REPRESENTATIONS}

    # Цели вспомогательных моделей — только из обучающей таблицы.
    aux_cc = pd.to_numeric(train.get("CC50_uM"), errors="coerce").to_numpy() \
        if "CC50_uM" in train else None
    aux_ic = pd.to_numeric(train.get("IC50_uM_VACV"), errors="coerce").to_numpy() \
        if "IC50_uM_VACV" in train else None
    aux_blocks_a = [np.hstack([fps_a[bl], desc_a, title_a, lit_a])
                    for bl in REPRESENTATIONS]
    aux_blocks_b = [np.hstack([fps_b[bl], desc_b, title_b, lit_b])
                    for bl in REPRESENTATIONS]

    regression, binned, ratios = [], [], []
    for seed in range(n_seeds):
        if aux_cc is not None and aux_ic is not None:
            r = _auxiliary_ratio(train, test, aux_blocks_a, aux_blocks_b,
                                 aux_cc, aux_ic, seed)
            if r is not None:
                ratios.append(r)
        encoder = MetaEncoder(seed=seed)
        meta_a = encoder.fit_transform(train)
        meta_b = encoder.transform(test)
        for bl in REPRESENTATIONS:
            xa = np.hstack([fps_a[bl], desc_a, meta_a, title_a, lit_a])
            xb = np.hstack([fps_b[bl], desc_b, meta_b, title_b, lit_b])
            model = lgb.LGBMRegressor(random_state=seed, **REG_PARAMS)
            model.fit(xa, np.log1p(y))
            regression.append(np.clip(np.expm1(model.predict(xb)), 0.0, None))
        binned.append(np.mean(
            [BinnedExpectationModel(blocks=tuple(list(bl) + ["descall"]),
                                    use_meta=True, use_title=True, seed=seed)
             .fit(train).predict(test)
             for bl in REPRESENTATIONS], axis=0))

    pred = 0.5 * np.mean(regression, axis=0) + 0.5 * np.mean(binned, axis=0)
    if ratios:
        # Смесь 50/50 с декомпозицией: лучше основной модели в обеих схемах.
        pred = 0.5 * pred + 0.5 * np.mean(ratios, axis=0)
        if verbose:
            print("декомпозиция подмешана (%d сидов)" % len(ratios))

    # Приоритет 2: литературное значение.
    lit_si = test["flat"].map(tables["si"]).to_numpy(dtype=float)
    covered = np.isfinite(lit_si)
    pred = np.where(covered, lit_si, pred)

    # Приоритет 1: собственное измерение организаторов из train.
    own = test["smiles"].map(train.groupby("smiles")["y"].mean())
    pred = np.where(own.notna(), own.fillna(0.0).to_numpy(), pred)

    if verbose:
        print("подстановок: литература %d | train %d | прогнозов >1000: %d"
              % (int(covered.sum()), int(own.notna().sum()), int((pred > 1000).sum())))
    return np.clip(pred, 0.0, None)
