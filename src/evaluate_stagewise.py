from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .metrics import (
    Stage1Metrics,
    compute_stage1_metrics,
    compute_multiclass_summary,
    confusion_matrix_for_labels,
    per_class_metrics_df,
)


def evaluate_stage1(
    y_true_attack: np.ndarray,
    p_attack: np.ndarray,
    threshold: float,
) -> Tuple[Stage1Metrics, np.ndarray, np.ndarray]:
    metrics = compute_stage1_metrics(y_true_attack=y_true_attack, p_attack=p_attack, threshold=threshold)
    y_pred_attack = (p_attack >= threshold).astype(int)
    cm = confusion_matrix(y_true_attack, y_pred_attack, labels=[0, 1])
    return metrics, y_pred_attack, cm


def evaluate_stage2_attack_only(
    y_true_attack_class: np.ndarray,
    y_pred_attack_class: np.ndarray,
    attack_labels: List[str],
) -> Dict[str, Any]:
    summary = compute_multiclass_summary(
        y_true_attack_class,
        y_pred_attack_class,
        labels=attack_labels,
    )
    per_class = per_class_metrics_df(y_true_attack_class, y_pred_attack_class, labels=attack_labels)
    cm = confusion_matrix_for_labels(y_true_attack_class, y_pred_attack_class, labels=attack_labels)

    return {
        "summary": summary,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
