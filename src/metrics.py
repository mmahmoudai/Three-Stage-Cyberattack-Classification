from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Stage1Metrics:
    roc_auc: float
    pr_auc: float
    precision_attack: float
    recall_attack: float
    f1_attack: float


def compute_stage1_metrics(y_true_attack: np.ndarray, p_attack: np.ndarray, threshold: float = 0.5) -> Stage1Metrics:
    roc = float(roc_auc_score(y_true_attack, p_attack))
    pr = float(average_precision_score(y_true_attack, p_attack))
    y_pred = (p_attack >= threshold).astype(int)
    prec = float(precision_score(y_true_attack, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true_attack, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true_attack, y_pred, pos_label=1, zero_division=0))
    return Stage1Metrics(roc_auc=roc, pr_auc=pr, precision_attack=prec, recall_attack=rec, f1_attack=f1)


def compute_multiclass_summary(y_true, y_pred, labels: List[str]) -> Dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def per_class_metrics_df(y_true, y_pred, labels: List[str]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for lab in labels:
        r = report.get(lab, None)
        if r is None:
            continue
        rows.append(
            {
                "class": lab,
                "precision": r.get("precision", 0.0),
                "recall": r.get("recall", 0.0),
                "f1": r.get("f1-score", 0.0),
                "support": r.get("support", 0),
            }
        )
    return pd.DataFrame(rows)


def confusion_pairs(cm: np.ndarray, labels: List[str], top_k: int = 10) -> List[Dict[str, object]]:
    out = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            out.append({"true": labels[i], "pred": labels[j], "count": int(cm[i, j])})
    out.sort(key=lambda d: d["count"], reverse=True)
    return [d for d in out[:top_k] if d["count"] > 0]


def confusion_matrix_for_labels(y_true, y_pred, labels: List[str]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
