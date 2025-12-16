from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    out_path: str,
    title: str,
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = False,
    annot: bool = True,
) -> None:
    """Save confusion matrix heatmap. If normalize=True, row-normalize to percentages."""
    if normalize:
        data = cm.astype(float)
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div by zero
        data = (data / row_sums) * 100
        fmt = ".1f"
    else:
        data = cm.astype(int)
        fmt = "d"
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        annot=annot and len(labels) <= 10,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count" if not normalize else "Recall %"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_class_distribution(
    df: pd.DataFrame,
    class_col: str,
    out_path: str,
    figsize: Tuple[int, int] = (12, 6),
    log_scale: bool = False,
) -> None:
    counts = df[class_col].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(counts)), counts.values, color="steelblue")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count" + (" (log scale)" if log_scale else ""))
    ax.set_title("Class Distribution of the Dataset")
    if log_scale:
        ax.set_yscale("log")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}",
                ha="center", va="bottom", fontsize=7, rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_architecture_figure(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    def box(x, y, w, h, text):
        r = matplotlib.patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="black",
            facecolor="#F2F2F2",
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    box(0.05, 0.35, 0.2, 0.3, "Input Flow\nFeatures")
    box(0.32, 0.35, 0.22, 0.3, "Stage 1\nAttack Gate\n(LightGBM)")
    box(0.62, 0.35, 0.22, 0.3, "Stage 2\nAttack Type\n(CatBoost)")
    box(0.88, 0.35, 0.1, 0.3, "Output\nClass")

    ax.annotate("", xy=(0.32, 0.5), xytext=(0.25, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.62, 0.5), xytext=(0.54, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.88, 0.5), xytext=(0.84, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.text(0.46, 0.2, "If benign: output Benign\nElse: pass to Stage 2", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_feature_importance(
    feature_names: Sequence[str],
    importances: Sequence[float],
    out_path: str,
    top_k: int = 20,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    df = pd.DataFrame({"feature": list(feature_names), "importance": list(importances)})
    df = df.sort_values("importance", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, y="feature", x="importance", ax=ax)
    ax.set_title("Top Feature Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_macro_f1_ci(
    macro_f1: float,
    ci_low: float,
    ci_high: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar([0], [macro_f1], yerr=[[macro_f1 - ci_low], [ci_high - macro_f1]], fmt="o", capsize=8, markersize=10, color="steelblue")
    ax.axhline(macro_f1, color="gray", linestyle="--", alpha=0.5)
    ax.text(0.15, macro_f1, f"{macro_f1:.4f}", va="center", fontsize=10)
    ax.text(0.15, ci_low, f"{ci_low:.4f}", va="top", fontsize=9, color="gray")
    ax.text(0.15, ci_high, f"{ci_high:.4f}", va="bottom", fontsize=9, color="gray")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Macro-F1")
    ax.set_title("End-to-End Macro-F1 with 95% CI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_error_analysis_chart(
    top_confusions: List[dict],
    lowest_recall_classes: List[dict],
    out_path: str,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Generate error analysis visualization with top confusion pairs and lowest recall classes."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Top confusion pairs
    ax1 = axes[0]
    if top_confusions:
        labels = [f"{c['true']} → {c['pred']}" for c in top_confusions[:10]]
        counts = [c["count"] for c in top_confusions[:10]]
        y_pos = range(len(labels))
        bars = ax1.barh(y_pos, counts, color="coral")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("Misclassification Count")
        ax1.set_title("Top 10 Confusion Pairs")
        for bar, val in zip(bars, counts):
            ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, str(val),
                     va="center", fontsize=8)

    # Right: Lowest recall attack classes
    ax2 = axes[1]
    if lowest_recall_classes:
        labels = [c["class"] for c in lowest_recall_classes]
        recalls = [c["recall"] * 100 for c in lowest_recall_classes]
        f1s = [c["f1"] * 100 for c in lowest_recall_classes]
        x_pos = np.arange(len(labels))
        width = 0.35
        bars1 = ax2.bar(x_pos - width/2, recalls, width, label="Recall %", color="steelblue")
        bars2 = ax2.bar(x_pos + width/2, f1s, width, label="F1 %", color="orange")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Percentage")
        ax2.set_title("Lowest Recall Attack Classes")
        ax2.legend(loc="upper right")
        ax2.set_ylim(0, 100)
        for bar in bars1:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{bar.get_height():.1f}", ha="center", fontsize=7)
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{bar.get_height():.1f}", ha="center", fontsize=7)

    fig.suptitle("Error Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_model_comparison_chart(
    metrics_df: pd.DataFrame,
    out_path: str,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """Generate model comparison bar chart for macro_f1, weighted_f1, accuracy."""
    models = metrics_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width, metrics_df["macro_f1"] * 100, width, label="Macro-F1 %", color="steelblue")
    bars2 = ax.bar(x, metrics_df["weighted_f1"] * 100, width, label="Weighted-F1 %", color="orange")
    bars3 = ax.bar(x + width, metrics_df["accuracy"] * 100, width, label="Accuracy %", color="green")

    ax.set_ylabel("Percentage")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)

    def add_labels(bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", fontsize=7)
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_per_class_f1_chart(
    per_class_df: pd.DataFrame,
    out_path: str,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Generate per-class F1 bar chart sorted by F1 score."""
    df_sorted = per_class_df.sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["coral" if f < 0.5 else "steelblue" for f in df_sorted["f1"]]
    bars = ax.barh(range(len(df_sorted)), df_sorted["f1"] * 100, color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["class"], fontsize=8)
    ax.set_xlabel("F1 Score (%)")
    ax.set_title("Per-Class F1 Scores (End-to-End)")
    ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend(loc="lower right")
    for bar, val in zip(bars, df_sorted["f1"] * 100):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=7)
    ax.set_xlim(0, 110)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
