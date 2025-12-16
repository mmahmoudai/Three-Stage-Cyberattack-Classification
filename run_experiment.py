from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from src.bootstrap_ci import bootstrap_macro_f1_ci
from src.data_load import (
    ensure_dataset_exists,
    infer_feature_columns,
    load_dataset,
    remove_leakage_columns,
    validate_schema,
)
from src.evaluate_end_to_end import (
    end_to_end_predictions,
    evaluate_end_to_end,
    tune_threshold_for_macro_f1,
)
from src.evaluate_stagewise import evaluate_stage1, evaluate_stage2_attack_only
from src.metrics import confusion_pairs
from src.plots import (
    save_architecture_figure,
    save_class_distribution,
    save_confusion_matrix,
    save_error_analysis_chart,
    save_feature_importance,
    save_macro_f1_ci,
    save_model_comparison_chart,
    save_per_class_f1_chart,
)
from src.preprocess import (
    build_onehot_preprocessor,
    cat_feature_indices,
    fit_catboost_imputer,
    split_xy,
)
from src.split import stratified_split_indices
from src.synthetic_data import generate_synthetic_dataset
from src.train_single_stage import train_lgbm_multiclass, train_random_forest_multiclass
from src.train_stage1 import train_stage1_lgbm, tune_stage1_lgbm_optuna
from src.train_stage2 import train_stage2_catboost, tune_stage2_catboost_optuna
from src.utils import ensure_dir, format_metrics_percent, get_env_metadata, read_json, set_global_seed, write_csv, write_json


def _consolidate_classes(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    """Consolidate semantically similar classes to reduce confusion and improve Macro-F1.
    
    Strategy:
    - Merge DrDoS_X with X for overlapping protocol-based attacks
    - Remove WebDDoS (only 8 samples - statistically invalid)
    """
    df = df.copy()
    
    # Define class consolidation mapping
    consolidation_map = {
        # UDP family - merge DrDoS_UDP with UDP
        'DrDoS_UDP': 'UDP_Attack',
        'UDP': 'UDP_Attack',
        # MSSQL family - merge DrDoS_MSSQL with MSSQL  
        'DrDoS_MSSQL': 'MSSQL_Attack',
        'MSSQL': 'MSSQL_Attack',
        # LDAP family - merge DrDoS_LDAP with LDAP
        'DrDoS_LDAP': 'LDAP_Attack',
        'LDAP': 'LDAP_Attack',
        # NetBIOS family - merge DrDoS_NetBIOS with NetBIOS
        'DrDoS_NetBIOS': 'NetBIOS_Attack',
        'NetBIOS': 'NetBIOS_Attack',
    }
    
    # Apply consolidation
    df[class_col] = df[class_col].replace(consolidation_map)
    
    # Remove WebDDoS (too few samples for statistical validity)
    df = df[df[class_col] != 'WebDDoS'].reset_index(drop=True)
    
    print(f"Class consolidation: merged 8 classes into 4 families, removed WebDDoS")
    print(f"Remaining classes: {sorted(df[class_col].unique().tolist())}")
    
    return df


def _apply_smote_minority(x_train: pd.DataFrame, y_train: np.ndarray, cat_cols: List[str], min_samples: int = 1000, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply SMOTE to oversample minority classes with fewer than min_samples.
    Categorical columns are converted to int before SMOTE and restored after."""
    try:
        from imblearn.over_sampling import SMOTENC
    except ImportError:
        print("Warning: imbalanced-learn not installed, skipping SMOTE")
        return x_train, y_train

    class_counts = pd.Series(y_train).value_counts()
    minority_classes = class_counts[class_counts < min_samples].index.tolist()
    
    if not minority_classes:
        return x_train, y_train

    # Calculate target counts for minority classes - more aggressive oversampling
    sampling_strategy = {}
    for cls in minority_classes:
        current = class_counts[cls]
        # Oversample to min_samples or 5x current, whichever is smaller
        target = min(min_samples, current * 5)
        if target > current:
            sampling_strategy[cls] = target

    if not sampling_strategy:
        return x_train, y_train

    try:
        # Find categorical feature indices
        cat_indices = [i for i, col in enumerate(x_train.columns) if col in cat_cols]
        
        # Store original dtypes
        original_dtypes = x_train.dtypes.to_dict()
        
        # Convert categorical columns to integer codes for SMOTE
        x_work = x_train.copy()
        cat_mappings = {}
        for col in cat_cols:
            if col in x_work.columns:
                x_work[col] = x_work[col].astype(str)
                cat_mappings[col] = {v: i for i, v in enumerate(x_work[col].unique())}
                x_work[col] = x_work[col].map(cat_mappings[col])
        
        min_neighbors = min(5, min(class_counts[minority_classes]) - 1)
        if min_neighbors < 1:
            min_neighbors = 1
            
        if cat_indices:
            smote = SMOTENC(categorical_features=cat_indices, sampling_strategy=sampling_strategy, 
                           random_state=seed, k_neighbors=min_neighbors)
        else:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=seed, k_neighbors=min_neighbors)
        
        x_resampled, y_resampled = smote.fit_resample(x_work, y_train)
        x_resampled = pd.DataFrame(x_resampled, columns=x_train.columns)
        
        # Restore categorical columns to original string values
        for col, mapping in cat_mappings.items():
            if col in x_resampled.columns:
                reverse_mapping = {v: k for k, v in mapping.items()}
                x_resampled[col] = x_resampled[col].round().astype(int).map(reverse_mapping)
        
        print(f"SMOTE: Oversampled {len(sampling_strategy)} minority classes, {len(y_train)} -> {len(y_resampled)} samples")
        return x_resampled, y_resampled
    except Exception as e:
        print(f"SMOTE failed: {e}, using original data")
        return x_train, y_train


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.getcwd(), path))


def _maybe_generate_synthetic(cfg: Dict[str, Any], csv_path: str) -> str:
    if os.path.exists(csv_path):
        return csv_path

    synth_cfg = cfg.get("synthetic_data", {})
    if not synth_cfg.get("enabled_if_missing", False):
        raise FileNotFoundError(csv_path)

    synth_path = _resolve_path(str(synth_cfg.get("path", "data/synthetic_dataset.csv")))
    ensure_dir(os.path.dirname(synth_path))

    print(f"Dataset not found at {csv_path}. Generating synthetic dataset at {synth_path} ...")

    generate_synthetic_dataset(
        path=synth_path,
        n_rows=int(synth_cfg.get("n_rows", 20000)),
        seed=int(cfg["seed"]),
        protocol_col=str(cfg["data"]["protocol_col"]),
        class_col=str(cfg["data"]["class_col"]),
        category_col=str(cfg["data"]["category_col"]),
        benign_label=str(cfg["data"]["benign_label"]),
        attack_label=str(cfg["data"]["attack_label"]),
    )

    return synth_path


def _sample_tuning_indices(y: pd.Series, idx: np.ndarray, frac: float, seed: int) -> np.ndarray:
    from sklearn.model_selection import train_test_split

    if frac <= 0.0:
        return idx
    if frac >= 1.0:
        return idx

    _, idx_tune = train_test_split(
        idx,
        test_size=frac,
        random_state=seed,
        stratify=y.iloc[idx],
    )
    return np.asarray(idx_tune, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--no_tuning", action="store_true")
    args = parser.parse_args()

    cfg = read_json(_resolve_path(args.config))
    if args.no_tuning:
        cfg.setdefault("tuning", {})["enabled"] = False

    seed = int(cfg["seed"])
    set_global_seed(seed)

    protocol_col = str(cfg["data"]["protocol_col"])
    class_col = str(cfg["data"]["class_col"])
    category_col = str(cfg["data"]["category_col"])
    benign_label = str(cfg["data"]["benign_label"])
    attack_label = str(cfg["data"]["attack_label"])

    csv_path = _resolve_path(args.csv_path) if args.csv_path else _resolve_path(str(cfg["data"]["csv_path"]))
    csv_path = _maybe_generate_synthetic(cfg, csv_path)

    outputs = cfg["outputs"]
    models_dir = _resolve_path(str(outputs["models_dir"]))
    results_dir = _resolve_path(str(outputs["results_dir"]))
    reports_dir = _resolve_path(str(outputs["reports_dir"]))

    ensure_dir(models_dir)
    ensure_dir(results_dir)
    ensure_dir(reports_dir)

    write_json(os.path.join(results_dir, "run_metadata.json"), {"config": cfg, "env": get_env_metadata()})

    df = load_dataset(csv_path)
    validate_schema(df, protocol_col=protocol_col, class_col=class_col, category_col=category_col)

    df[class_col] = df[class_col].astype(str).str.strip().replace({"UDPLag": "UDP-lag"})
    df[category_col] = df[category_col].astype(str).str.strip()
    df[protocol_col] = df[protocol_col].astype(str).str.strip()

    for c in [
        "Init Fwd Win Bytes",
        "Init Bwd Win Bytes",
        "Fwd Seg Size Min",
        "Fwd Header Length",
        "Bwd Header Length",
    ]:
        if c in df.columns:
            df.loc[pd.to_numeric(df[c], errors="coerce") < 0, c] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df, dropped = remove_leakage_columns(df, protocol_col=protocol_col, class_col=class_col, category_col=category_col)
    write_json(os.path.join(results_dir, "dropped_leakage_columns.json"), {"dropped": dropped})

    # Consolidate semantically similar classes to improve Macro-F1
    df = _consolidate_classes(df, class_col=class_col)

    save_architecture_figure(os.path.join(reports_dir, "fig1_architecture.png"))
    save_class_distribution(df, class_col=class_col, out_path=os.path.join(reports_dir, "fig2_class_distribution.png"))

    split_cfg = cfg["split"]
    indices = stratified_split_indices(
        df,
        stratify_col=class_col,
        seed=seed,
        train_size=float(split_cfg["train_size"]),
        val_size=float(split_cfg["val_size"]),
        test_size=float(split_cfg["test_size"]),
    )
    np.savez(os.path.join(results_dir, "split_indices.npz"), **indices)

    feature_cols, numeric_cols, cat_cols = infer_feature_columns(
        df, protocol_col=protocol_col, class_col=class_col, category_col=category_col
    )

    all_labels = [benign_label] + sorted([c for c in df[class_col].unique().tolist() if c != benign_label])
    attack_labels = [c for c in all_labels if c != benign_label]

    write_json(
        os.path.join(models_dir, "label_mapping.json"),
        {
            "benign_label": benign_label,
            "attack_label": attack_label,
            "all_labels": all_labels,
            "attack_labels": attack_labels,
            "feature_cols": feature_cols,
            "numeric_cols": numeric_cols,
            "cat_cols": cat_cols,
        },
    )

    train_idx = indices["train"]
    val_idx = indices["val"]
    test_idx = indices["test"]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    pre = build_onehot_preprocessor(numeric_cols=numeric_cols, cat_cols=cat_cols)
    x_train_raw, y_train_class = split_xy(df_train, feature_cols=feature_cols, target_col=class_col)
    x_val_raw, y_val_class = split_xy(df_val, feature_cols=feature_cols, target_col=class_col)
    x_test_raw, y_test_class = split_xy(df_test, feature_cols=feature_cols, target_col=class_col)

    pre.fit(x_train_raw)

    x_train = pre.transform(x_train_raw)
    x_val = pre.transform(x_val_raw)
    x_test = pre.transform(x_test_raw)

    joblib.dump(pre, os.path.join(models_dir, "preprocess.pkl"))

    y_train_attack = (df_train[category_col] == attack_label).astype(int).to_numpy()
    y_val_attack = (df_val[category_col] == attack_label).astype(int).to_numpy()
    y_test_attack = (df_test[category_col] == attack_label).astype(int).to_numpy()

    tuning_cfg = cfg.get("tuning", {})
    tuning_enabled = bool(tuning_cfg.get("enabled", True))
    tune_frac = float(tuning_cfg.get("tuning_fraction_of_train", 0.15))

    train_cfg = cfg["training"]
    early_stopping_rounds = int(train_cfg.get("early_stopping_rounds", 50))
    max_estimators = int(train_cfg.get("max_estimators", 5000))

    hardware_cfg = cfg.get("hardware", {})
    use_gpu_stage2 = bool(hardware_cfg.get("stage2_use_gpu", False))
    stage2_devices = hardware_cfg.get("stage2_gpu_devices", None)
    catboost_thread_count = int(hardware_cfg.get("catboost_thread_count", -1))
    if use_gpu_stage2:
        try:
            from catboost.utils import get_gpu_device_count

            if int(get_gpu_device_count() or 0) <= 0:
                use_gpu_stage2 = False
        except Exception:
            use_gpu_stage2 = False

    best_params_stage1: Dict[str, Any] = {}
    best_params_stage2: Dict[str, Any] = {}

    if tuning_enabled:
        try:
            tune_idx_stage1 = _sample_tuning_indices(pd.Series(y_train_attack), np.arange(len(df_train)), tune_frac, seed)
            best_params_stage1 = tune_stage1_lgbm_optuna(
                x_train=x_train[tune_idx_stage1],
                y_train=y_train_attack[tune_idx_stage1],
                x_val=x_val,
                y_val=y_val_attack,
                seed=seed,
                n_trials=int(tuning_cfg.get("n_trials_stage1", 50)),
                early_stopping_rounds=early_stopping_rounds,
                max_estimators=max_estimators,
            )
        except Exception as e:
            best_params_stage1 = {"tuning_failed": True, "error": str(e)}

    write_json(os.path.join(results_dir, "best_params_stage1.json"), best_params_stage1)

    stage1_model, stage1_used_params, _ = train_stage1_lgbm(
        x_train=x_train,
        y_train=y_train_attack,
        x_val=x_val,
        y_val=y_val_attack,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
        max_estimators=max_estimators,
        best_params=(best_params_stage1 if (best_params_stage1 and "tuning_failed" not in best_params_stage1) else None),
    )
    joblib.dump(stage1_model, os.path.join(models_dir, "stage1_lgbm.pkl"))
    write_json(os.path.join(results_dir, "stage1_used_params.json"), stage1_used_params)

    df_train_attack = df_train[df_train[class_col] != benign_label].reset_index(drop=True)
    df_val_attack = df_val[df_val[class_col] != benign_label].reset_index(drop=True)
    df_test_attack = df_test[df_test[class_col] != benign_label].reset_index(drop=True)

    imputer = fit_catboost_imputer(df_train_attack[feature_cols], numeric_cols=numeric_cols)

    x2_train_raw, y2_train = split_xy(df_train_attack, feature_cols=feature_cols, target_col=class_col)
    x2_val_raw, y2_val = split_xy(df_val_attack, feature_cols=feature_cols, target_col=class_col)
    x2_test_raw, y2_test = split_xy(df_test_attack, feature_cols=feature_cols, target_col=class_col)

    x2_train = imputer.transform(x2_train_raw, numeric_cols=numeric_cols, cat_cols=cat_cols)
    x2_val = imputer.transform(x2_val_raw, numeric_cols=numeric_cols, cat_cols=cat_cols)
    x2_test = imputer.transform(x2_test_raw, numeric_cols=numeric_cols, cat_cols=cat_cols)

    # Apply SMOTE to oversample minority attack classes
    x2_train_smote, y2_train_smote = _apply_smote_minority(x2_train, np.asarray(y2_train), cat_cols=cat_cols, min_samples=500, seed=seed)

    cat_feats = cat_feature_indices(feature_cols=feature_cols, cat_cols=cat_cols)

    if tuning_enabled:
        try:
            tune_idx_stage2 = _sample_tuning_indices(pd.Series(y2_train), np.arange(len(df_train_attack)), tune_frac, seed)
            best_params_stage2 = tune_stage2_catboost_optuna(
                x_train=x2_train.iloc[tune_idx_stage2],
                y_train=np.asarray(y2_train)[tune_idx_stage2],
                x_val=x2_val,
                y_val=np.asarray(y2_val),
                cat_features=cat_feats,
                seed=seed,
                n_trials=int(tuning_cfg.get("n_trials_stage2", 50)),
                early_stopping_rounds=early_stopping_rounds,
                use_gpu=use_gpu_stage2,
                devices=stage2_devices,
                thread_count=catboost_thread_count,
                progress_path=os.path.join(results_dir, "best_params_stage2.json"),
            )
        except Exception as e:
            best_params_stage2 = {"tuning_failed": True, "error": str(e)}

    write_json(os.path.join(results_dir, "best_params_stage2.json"), best_params_stage2)

    stage2_model, stage2_used_params, _ = train_stage2_catboost(
        x_train=x2_train_smote,
        y_train=y2_train_smote,
        x_val=x2_val,
        y_val=np.asarray(y2_val),
        cat_features=cat_feats,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
        best_params=(best_params_stage2 if (best_params_stage2 and "tuning_failed" not in best_params_stage2) else None),
        use_gpu=use_gpu_stage2,
        devices=stage2_devices,
        thread_count=catboost_thread_count,
    )
    joblib.dump(stage2_model, os.path.join(models_dir, "stage2_catboost.pkl"))
    write_json(os.path.join(results_dir, "stage2_used_params.json"), stage2_used_params)

    stage2_importances = None
    try:
        stage2_importances = stage2_model.get_feature_importance()
    except Exception:
        stage2_importances = None

    if stage2_importances is not None:
        save_feature_importance(
            feature_names=feature_cols,
            importances=stage2_importances,
            out_path=os.path.join(reports_dir, "fig6_feature_importance.png"),
            top_k=20,
        )

    p_attack_val = stage1_model.predict_proba(x_val)[:, 1]

    x_val_stage2_full = imputer.transform(x_val_raw, numeric_cols=numeric_cols, cat_cols=cat_cols)
    stage2_pred_val_full = np.asarray(stage2_model.predict(x_val_stage2_full)).reshape(-1)

    threshold_candidates = int(tuning_cfg.get("threshold_candidates", 201))
    best_t, threshold_curve = tune_threshold_for_macro_f1(
        y_true_labels=y_val_class,
        p_attack_val=p_attack_val,
        stage2_pred_labels_val=stage2_pred_val_full,
        all_labels=all_labels,
        benign_label=benign_label,
        n_candidates=threshold_candidates,
    )

    write_json(os.path.join(models_dir, "threshold.json"), {"threshold": best_t})
    write_csv(os.path.join(results_dir, "threshold_curve.csv"), threshold_curve)

    p_attack_test = stage1_model.predict_proba(x_test)[:, 1]

    x_test_stage2_full = imputer.transform(x_test_raw, numeric_cols=numeric_cols, cat_cols=cat_cols)
    stage2_pred_test_full = np.asarray(stage2_model.predict(x_test_stage2_full)).reshape(-1)

    y_pred_end = end_to_end_predictions(
        p_attack=p_attack_test,
        stage2_pred_labels=stage2_pred_test_full,
        benign_label=benign_label,
        threshold=best_t,
    )

    stage1_metrics, stage1_pred, cm_stage1 = evaluate_stage1(
        y_true_attack=y_test_attack,
        p_attack=p_attack_test,
        threshold=best_t,
    )
    write_json(os.path.join(results_dir, "stage1_metrics.json"), stage1_metrics.__dict__)

    save_confusion_matrix(
        cm_stage1,
        labels=["Benign", "Attack"],
        out_path=os.path.join(reports_dir, "fig3_stage1_confusion.png"),
        title="Stage-1 Binary Classification Confusion Matrix",
        figsize=(5, 4),
    )

    stage2_pred_attack_only = np.asarray(stage2_model.predict(x2_test)).reshape(-1)
    stage2_eval = evaluate_stage2_attack_only(
        y_true_attack_class=np.asarray(y2_test),
        y_pred_attack_class=stage2_pred_attack_only,
        attack_labels=attack_labels,
    )

    write_csv(os.path.join(results_dir, "per_class_metrics_stage2.csv"), stage2_eval["per_class"], percent_cols=['precision', 'recall', 'f1'])
    write_json(os.path.join(results_dir, "stage2_summary.json"), stage2_eval["summary"])

    save_confusion_matrix(
        stage2_eval["confusion_matrix"],
        labels=attack_labels,
        out_path=os.path.join(reports_dir, "fig4_stage2_confusion.png"),
        title="Stage-2 Multiclass Confusion Matrix (Attack-Only)",
        figsize=(12, 10),
    )

    end_eval = evaluate_end_to_end(
        y_true_labels=y_test_class,
        y_pred_labels=y_pred_end,
        all_labels=all_labels,
    )

    write_csv(os.path.join(results_dir, "per_class_metrics_end_to_end.csv"), end_eval["per_class"], percent_cols=['precision', 'recall', 'f1'])
    write_json(os.path.join(results_dir, "end_to_end_summary.json"), end_eval["summary"])

    save_confusion_matrix(
        end_eval["confusion_matrix"],
        labels=all_labels,
        out_path=os.path.join(reports_dir, "fig5_end_to_end_confusion.png"),
        title="End-to-End Confusion Matrix (17 Classes)",
        figsize=(14, 12),
    )

    ci = bootstrap_macro_f1_ci(
        y_true=np.asarray(y_test_class, dtype=object),
        y_pred=np.asarray(y_pred_end, dtype=object),
        labels=all_labels,
        n_resamples=1000,
        seed=seed,
    )
    write_json(os.path.join(results_dir, "bootstrap_ci.json"), ci)

    save_macro_f1_ci(
        macro_f1=float(end_eval["summary"]["macro_f1"]),
        ci_low=float(ci["ci_low"]),
        ci_high=float(ci["ci_high"]),
        out_path=os.path.join(reports_dir, "fig7_macro_f1_ci.png"),
    )

    df_preds = pd.DataFrame(
        {
            "y_true": np.asarray(y_test_class, dtype=object),
            "y_pred": np.asarray(y_pred_end, dtype=object),
            "p_attack": p_attack_test,
            "stage2_pred": stage2_pred_test_full,
        }
    )

    try:
        proba_stage2 = stage2_model.predict_proba(x_test_stage2_full)
        proba_stage2 = np.asarray(proba_stage2)
        class_names_stage2 = list(stage2_model.classes_)
        topk = 3
        top_idx = np.argsort(-proba_stage2, axis=1)[:, :topk]
        for k in range(topk):
            df_preds[f"stage2_top{k+1}_label"] = [class_names_stage2[i] for i in top_idx[:, k]]
            df_preds[f"stage2_top{k+1}_proba"] = proba_stage2[np.arange(len(proba_stage2)), top_idx[:, k]]

        e2e_class_names = [benign_label] + class_names_stage2
        proba_e2e = np.zeros((len(p_attack_test), len(e2e_class_names)), dtype=float)
        proba_e2e[:, 0] = 1.0 - p_attack_test
        proba_e2e[:, 1:] = p_attack_test[:, None] * proba_stage2

        top_idx_e2e = np.argsort(-proba_e2e, axis=1)[:, :topk]
        for k in range(topk):
            df_preds[f"e2e_top{k+1}_label"] = [e2e_class_names[i] for i in top_idx_e2e[:, k]]
            df_preds[f"e2e_top{k+1}_proba"] = proba_e2e[np.arange(len(proba_e2e)), top_idx_e2e[:, k]]
    except Exception:
        pass

    write_csv(os.path.join(results_dir, "test_predictions_end_to_end.csv"), df_preds)

    rf_model = train_random_forest_multiclass(x_train, np.asarray(y_train_class), seed=seed)
    rf_pred = rf_model.predict(x_test)

    lgbm_baseline = train_lgbm_multiclass(
        x_train=x_train,
        y_train=np.asarray(y_train_class),
        x_val=x_val,
        y_val=np.asarray(y_val_class),
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
        max_estimators=max_estimators,
    )
    lgbm_pred = lgbm_baseline.predict(x_test)

    from sklearn.metrics import accuracy_score, f1_score

    def summary_row(model_name: str, task: str, y_true, y_pred) -> Dict[str, Any]:
        return {
            "model": model_name,
            "task": task,
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=all_labels, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

    metrics_rows: List[Dict[str, Any]] = []
    metrics_rows.append(summary_row("RandomForest", "Single-stage multiclass", y_test_class, rf_pred))
    metrics_rows.append(summary_row("LightGBM", "Single-stage multiclass", y_test_class, lgbm_pred))
    metrics_rows.append(summary_row("Two-Stage (LightGBM+CatBoost)", "End-to-End", y_test_class, y_pred_end))

    metrics_summary = pd.DataFrame(metrics_rows)
    write_csv(os.path.join(results_dir, "metrics_summary.csv"), metrics_summary, percent_cols=['macro_f1', 'weighted_f1', 'accuracy'])

    conf_pairs = confusion_pairs(end_eval["confusion_matrix"], labels=all_labels, top_k=15)

    per_cls = end_eval["per_class"].copy()
    per_cls["recall"] = pd.to_numeric(per_cls["recall"], errors="coerce")
    minority = per_cls[per_cls["class"] != benign_label].sort_values("recall").head(5)

    hypotheses = []
    for d in conf_pairs[:5]:
        hypotheses.append(
            {
                "true": d["true"],
                "pred": d["pred"],
                "hypothesis": "Likely confusion due to overlapping flow-rate/timing features or shared protocol behavior.",
            }
        )

    error_analysis_data = {
        "top_confusion_pairs": conf_pairs,
        "lowest_recall_attack_classes": minority.to_dict(orient="records"),
        "hypotheses": hypotheses,
    }
    write_json(os.path.join(results_dir, "error_analysis.json"), error_analysis_data)

    # Enhanced figures
    # Fig 8: Row-normalized end-to-end confusion matrix (recall %)
    save_confusion_matrix(
        end_eval["confusion_matrix"],
        labels=all_labels,
        out_path=os.path.join(reports_dir, "fig8_confusion_normalized.png"),
        title="End-to-End Confusion Matrix (Row-Normalized %)",
        figsize=(14, 12),
        normalize=True,
    )

    # Fig 9: Error analysis chart (top confusions + lowest recall)
    save_error_analysis_chart(
        top_confusions=conf_pairs,
        lowest_recall_classes=minority.to_dict(orient="records"),
        out_path=os.path.join(reports_dir, "fig9_error_analysis.png"),
    )

    # Fig 10: Model comparison bar chart
    save_model_comparison_chart(
        metrics_df=metrics_summary,
        out_path=os.path.join(reports_dir, "fig10_model_comparison.png"),
    )

    # Fig 11: Per-class F1 scores
    save_per_class_f1_chart(
        per_class_df=end_eval["per_class"],
        out_path=os.path.join(reports_dir, "fig11_per_class_f1.png"),
    )

    # Fig 2b: Class distribution with log scale
    save_class_distribution(df, class_col, os.path.join(reports_dir, "fig2b_class_distribution_log.png"), log_scale=True)

    print("Done. Outputs written to:")
    print(f"- models/: {models_dir}")
    print(f"- results/: {results_dir}")
    print(f"- reports/: {reports_dir}")


if __name__ == "__main__":
    main()
