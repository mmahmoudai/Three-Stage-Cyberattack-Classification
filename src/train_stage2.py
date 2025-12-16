from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score


def _require_catboost():
    try:
        import catboost  # noqa: F401
    except Exception as e:
        raise ImportError(
            "catboost is required. Install with: pip install catboost"
        ) from e


def train_stage2_catboost(
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    cat_features: Optional[list[int]],
    seed: int,
    early_stopping_rounds: int,
    best_params: Optional[Dict[str, Any]] = None,
    use_gpu: bool = False,
    devices: Optional[str] = None,
    thread_count: int = -1,
) -> Tuple[Any, Dict[str, Any], float]:
    _require_catboost()

    from catboost import CatBoostClassifier

    params: Dict[str, Any] = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "iterations": 7000,
        "learning_rate": 0.022,
        "depth": 10,
        "l2_leaf_reg": 0.4,
        "random_strength": 0.15,
        "bagging_temperature": 0.08,
        "auto_class_weights": "Balanced",
        "border_count": 200,
        "min_data_in_leaf": 4,
        "grow_policy": "Lossguide",
        "max_leaves": 72,
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": early_stopping_rounds,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": int(thread_count),
    }
    if best_params:
        params.update(best_params)

    if use_gpu:
        params["task_type"] = "GPU"
        if devices:
            params["devices"] = str(devices)

    model = CatBoostClassifier(**params)

    model.fit(
        x_train,
        y_train,
        cat_features=cat_features,
        eval_set=(x_val, y_val),
        use_best_model=True,
    )

    y_pred_val = model.predict(x_val)
    y_pred_val = np.asarray(y_pred_val).reshape(-1)

    labels = np.unique(
        np.concatenate(
            [
                np.asarray(y_train).reshape(-1),
                np.asarray(y_val).reshape(-1),
            ]
        )
    )
    macro_f1 = float(f1_score(y_val, y_pred_val, average="macro", labels=labels, zero_division=0))

    used_params = dict(params)
    used_params["best_iteration_"] = int(getattr(model, "get_best_iteration", lambda: 0)() or 0)
    return model, used_params, macro_f1


def tune_stage2_catboost_optuna(
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    cat_features: Optional[list[int]],
    seed: int,
    n_trials: int,
    early_stopping_rounds: int,
    use_gpu: bool = False,
    devices: Optional[str] = None,
    thread_count: int = -1,
    progress_path: Optional[str] = None,
) -> Dict[str, Any]:
    _require_catboost()

    try:
        import optuna
    except Exception as e:
        raise ImportError("optuna is required for optuna tuning") from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from catboost import CatBoostClassifier

    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1:average=Macro",
            "iterations": 6000,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.05, log=True),
            "depth": trial.suggest_int("depth", 8, 12),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 2.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 0.2),
            "border_count": 200,
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 3, 10),
            "grow_policy": "Lossguide",
            "max_leaves": trial.suggest_int("max_leaves", 48, 96),
            "auto_class_weights": "Balanced",
            "random_seed": seed,
            "od_type": "Iter",
            "od_wait": early_stopping_rounds,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": int(thread_count),
        }

        if use_gpu:
            params["task_type"] = "GPU"
            if devices:
                params["devices"] = str(devices)

        model = CatBoostClassifier(**params)
        model.fit(
            x_train,
            y_train,
            cat_features=cat_features,
            eval_set=(x_val, y_val),
            use_best_model=True,
        )

        y_pred_val = np.asarray(model.predict(x_val)).reshape(-1)
        labels = np.unique(
            np.concatenate(
                [
                    np.asarray(y_train).reshape(-1),
                    np.asarray(y_val).reshape(-1),
                ]
            )
        )
        return float(f1_score(y_val, y_pred_val, average="macro", labels=labels, zero_division=0))

    def _checkpoint(study: optuna.Study, _: optuna.Trial) -> None:
        if not progress_path:
            return
        try:
            best_params = dict(getattr(study, "best_params", {}) or {})
            tmp_path = progress_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2)
            os.replace(tmp_path, progress_path)
        except Exception:
            return

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_checkpoint])

    return dict(study.best_params)
