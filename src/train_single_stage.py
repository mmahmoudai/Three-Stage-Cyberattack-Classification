from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def _require_lightgbm():
    try:
        import lightgbm as lgb  # noqa: F401
    except Exception as e:
        raise ImportError(
            "lightgbm is required. Install with: pip install lightgbm"
        ) from e


def train_random_forest_multiclass(
    x_train,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)
    return model


def train_lgbm_multiclass(
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    seed: int,
    early_stopping_rounds: int,
    max_estimators: int,
) -> Any:
    _require_lightgbm()
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=max_estimators,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        deterministic=True,
        verbosity=-1,
    )

    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=callbacks,
    )

    return model


def predict_ids(model: Any, x) -> np.ndarray:
    return np.asarray(model.predict(x)).reshape(-1)


def multiclass_summary_from_ids(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(np.mean(y_true == y_pred)),
    }
