from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score


def _require_lightgbm():
    try:
        import lightgbm as lgb  # noqa: F401
    except Exception as e:
        raise ImportError(
            "lightgbm is required. Install with: pip install lightgbm"
        ) from e


def train_stage1_lgbm(
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    seed: int,
    early_stopping_rounds: int,
    max_estimators: int,
    best_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any], float]:
    _require_lightgbm()
    import lightgbm as lgb

    params = {
        "n_estimators": max_estimators,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "binary",
        "random_state": seed,
        "n_jobs": -1,
        "deterministic": True,
        "verbosity": -1,
    }
    if best_params:
        params.update(best_params)

    model = lgb.LGBMClassifier(**params)

    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="aucpr",
        callbacks=callbacks,
    )

    p_val = model.predict_proba(x_val)[:, 1]
    pr_auc = float(average_precision_score(y_val, p_val))

    used_params = dict(params)
    used_params["best_iteration_"] = int(getattr(model, "best_iteration_", 0) or 0)
    return model, used_params, pr_auc


def tune_stage1_lgbm_optuna(
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    seed: int,
    n_trials: int,
    early_stopping_rounds: int,
    max_estimators: int,
) -> Dict[str, Any]:
    _require_lightgbm()

    try:
        import optuna
    except Exception as e:
        raise ImportError("optuna is required for optuna tuning") from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    import lightgbm as lgb

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": max_estimators,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "objective": "binary",
            "random_state": seed,
            "n_jobs": -1,
            "deterministic": True,
            "verbosity": -1,
        }

        model = lgb.LGBMClassifier(**params)
        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="aucpr",
            callbacks=callbacks,
        )

        p_val = model.predict_proba(x_val)[:, 1]
        return float(average_precision_score(y_val, p_val))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return dict(study.best_params)
