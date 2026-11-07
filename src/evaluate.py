"""
Стенд оценки.

Две вещи, без которых на этой задаче легко обмануть себя:

1. MSE считается на ИСХОДНОМ масштабе SI — той же величине, что и на Kaggle.
   Оптимизация MSE в логарифме приводит к другому оптимуму, и улучшение там
   не означает улучшения на лидерборде.

2. Одно число ничего не значит. На 320 молекулах, где единственный объект
   с SI = 36067 даёт заметную часть ошибки, результат сильно зависит от того,
   в какой фолд он попал. Поэтому всё меряется повторами по нескольким сидам,
   и сравнивать изменения нужно с разбросом, а не с одним прогоном.
"""

import numpy as np
import pandas as pd

from splits import repeated_scaffold_kfold

SENTINEL = 10000.0


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def decompose_error(y_true, y_pred):
    """
    Разбивает суммарную MSE на вклад заглушек и активных молекул.
    Показывает, куда вкладывать усилия: в классификатор или в регрессор.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    is_sent = y_true == SENTINEL
    se = (y_true - y_pred) ** 2
    return {
        "mse": float(se.mean()),
        "share_sentinel": float(se[is_sent].sum() / se.sum()) if se.sum() > 0 else 0.0,
        "mse_on_sentinel": float(se[is_sent].mean()) if is_sent.any() else np.nan,
        "mse_on_active": float(se[~is_sent].mean()) if (~is_sent).any() else np.nan,
        "n_sentinel": int(is_sent.sum()),
        "n_active": int((~is_sent).sum()),
        # Вклад худших 1% объектов: показывает, насколько результат
        # держится на единичных выбросах
        "share_worst_1pct": float(np.sort(se)[-max(1, n // 100):].sum() / se.sum())
        if se.sum() > 0 else 0.0,
    }


def repeated_cv(fit_predict, df, n_splits=5, n_repeats=5, base_seed=0, verbose=True,
                label=""):
    """
    fit_predict(train_df, val_df, seed) -> np.ndarray предсказаний для val_df.

    Возвращает словарь с OOF MSE по каждому повтору, средним и разбросом.
    """
    smiles = df["smiles"].tolist()
    y = df["y"].to_numpy(dtype=float)

    per_repeat = []
    oof_last = None
    for seed, folds in repeated_scaffold_kfold(smiles, n_splits, n_repeats, base_seed):
        oof = np.full(len(df), np.nan)
        for tr_idx, va_idx in folds:
            pred = fit_predict(df.iloc[tr_idx], df.iloc[va_idx], seed)
            oof[va_idx] = np.asarray(pred, dtype=float)
        if np.isnan(oof).any():
            raise ValueError("не все объекты получили предсказание")
        per_repeat.append(mse(y, oof))
        oof_last = oof

    per_repeat = np.array(per_repeat)
    res = {
        "label": label,
        "mse_mean": float(per_repeat.mean()),
        "mse_std": float(per_repeat.std(ddof=1)) if len(per_repeat) > 1 else 0.0,
        "mse_min": float(per_repeat.min()),
        "mse_max": float(per_repeat.max()),
        "per_repeat": per_repeat.tolist(),
        "n_repeats": n_repeats,
        "n_splits": n_splits,
    }
    res.update({f"last_{k}": v for k, v in decompose_error(y, oof_last).items()})

    if verbose:
        report(res)
    return res


def report(res):
    lab = f"[{res['label']}] " if res.get("label") else ""
    print(f"{lab}OOF MSE = {res['mse_mean']:,.0f} ± {res['mse_std']:,.0f}"
          f"   (разброс {res['mse_min']:,.0f} … {res['mse_max']:,.0f},"
          f" {res['n_repeats']}×{res['n_splits']})")
    print(f"    вклад заглушек в ошибку: {res['last_share_sentinel']:.0%}"
          f"   |  MSE на заглушках {res['last_mse_on_sentinel']:,.0f}"
          f"   |  на активных {res['last_mse_on_active']:,.0f}")
    print(f"    доля ошибки от худшего 1% объектов: {res['last_share_worst_1pct']:.0%}")


def compare(res_a, res_b):
    """
    Сравнение двух вариантов с поправкой на шум. Улучшение считается
    значимым, только если оно больше объединённого разброса.
    """
    d = res_a["mse_mean"] - res_b["mse_mean"]
    pooled = np.hypot(res_a["mse_std"], res_b["mse_std"])
    verdict = "значимо" if abs(d) > pooled else "В ПРЕДЕЛАХ ШУМА"
    better = res_b["label"] if d > 0 else res_a["label"]
    print(f"разница {abs(d):,.0f} (разброс {pooled:,.0f}) → {verdict}"
          + (f", лучше: {better}" if verdict == "значимо" else ""))
    return {"delta": float(d), "pooled_std": float(pooled), "significant": abs(d) > pooled}


def make_submission(test_df, preds, path="submission.csv", clip_min=0.0, clip_max=None):
    """
    Сборка файла ответа. Формат из задания: index, SI VACV.
    Проверяет полноту и отсутствие недопустимых значений.
    """
    preds = np.asarray(preds, dtype=float)
    if len(preds) != len(test_df):
        raise ValueError(f"предсказаний {len(preds)}, строк в test {len(test_df)}")
    if np.isnan(preds).any():
        raise ValueError(f"NaN в предсказаниях: {int(np.isnan(preds).sum())}")
    preds = np.clip(preds, clip_min, clip_max if clip_max is not None else np.inf)

    sub = pd.DataFrame({"index": test_df["index"].to_numpy(), "SI VACV": preds})
    if sub["index"].duplicated().any():
        raise ValueError("повторяющиеся index")
    sub = sub.sort_values("index").reset_index(drop=True)
    sub.to_csv(path, index=False)
    print(f"записано {len(sub)} строк в {path}"
          f"   (min {preds.min():,.2f}, медиана {np.median(preds):,.2f}, max {preds.max():,.2f})")
    return sub
