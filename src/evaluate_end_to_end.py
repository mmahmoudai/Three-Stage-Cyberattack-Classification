from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    compute_multiclass_summary,
    confusion_matrix_for_labels,
    per_class_metrics_df,
)


def end_to_end_predictions(
    p_attack: np.ndarray,
    stage2_pred_labels: Sequence[str],
    benign_label: str,
    threshold: float,
) -> np.ndarray:
    stage2_pred_labels = np.asarray(stage2_pred_labels, dtype=object)
    out = np.full(shape=(len(p_attack),), fill_value=benign_label, dtype=object)
    m = p_attack >= threshold
    out[m] = stage2_pred_labels[m]
    return out


def tune_threshold_for_macro_f1(
    y_true_labels: Sequence[str],
    p_attack_val: np.ndarray,
    stage2_pred_labels_val: Sequence[str],
    all_labels: List[str],
    benign_label: str,
    n_candidates: int,
) -> Tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.0, 1.0, n_candidates)

    rows = []
    best_t = 0.5
    best_score = -1.0

    y_true = np.asarray(y_true_labels, dtype=object)

    for t in thresholds:
        y_pred = end_to_end_predictions(
            p_attack=p_attack_val,
            stage2_pred_labels=stage2_pred_labels_val,
            benign_label=benign_label,
            threshold=float(t),
        )
        summary = compute_multiclass_summary(y_true, y_pred, labels=all_labels)
        score = summary["macro_f1"]
        rows.append({"threshold": float(t), "macro_f1": float(score)})
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, pd.DataFrame(rows)


def evaluate_end_to_end(
    y_true_labels: Sequence[str],
    y_pred_labels: Sequence[str],
    all_labels: List[str],
) -> Dict[str, Any]:
    y_true = np.asarray(y_true_labels, dtype=object)
    y_pred = np.asarray(y_pred_labels, dtype=object)

    summary = compute_multiclass_summary(y_true, y_pred, labels=all_labels)
    per_class = per_class_metrics_df(y_true, y_pred, labels=all_labels)
    cm = confusion_matrix_for_labels(y_true, y_pred, labels=all_labels)

    return {
        "summary": summary,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
