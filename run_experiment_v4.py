"""
V4 Experiment Runner
Train and evaluate the Three-Stage Hierarchical Classification System.
Compare V4 vs V3 performance.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

from src.v4_pipeline import V4Pipeline, evaluate_v4_pipeline
from src.utils import write_json, set_global_seed


# Class consolidation mapping (same as V3)
CLASS_CONSOLIDATION = {
    "UDP": "UDP_Attack",
    "DrDoS_UDP": "UDP_Attack",
    "MSSQL": "MSSQL_Attack",
    "DrDoS_MSSQL": "MSSQL_Attack",
    "NetBIOS": "NetBIOS_Attack",
    "DrDoS_NetBIOS": "NetBIOS_Attack",
    "LDAP": "LDAP_Attack",
    "DrDoS_LDAP": "LDAP_Attack",
}

REMOVE_CLASSES = ["WebDDoS"]


def load_and_prepare_data(data_path: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load data, consolidate classes, and split into train/val/test."""
    
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples")
    
    # Get label column - use 'Class' which has specific attack types
    if "Class" in df.columns:
        label_col = "Class"
    elif "Label" in df.columns:
        label_col = "Label"
    else:
        label_col = df.columns[-1]
    
    print(f"  Using label column: {label_col}")
    
    # Consolidate classes
    print("Consolidating classes...")
    for old_class, new_class in CLASS_CONSOLIDATION.items():
        df.loc[df[label_col] == old_class, label_col] = new_class
    
    # Remove classes with too few samples
    for cls in REMOVE_CLASSES:
        df = df[df[label_col] != cls]
    
    class_counts = df[label_col].value_counts()
    print(f"  Classes after consolidation: {len(class_counts)}")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count}")
    
    # Separate features and labels
    X = df.drop(columns=[label_col])
    y = df[label_col].values
    
    # Handle infinite and missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Remove leakage and non-numeric columns
    leakage_cols = [c for c in X.columns if any(x in c.lower() for x in ['flow id', 'source ip', 'destination ip', 'timestamp', 'src ip', 'dst ip', 'class'])]
    
    # Also remove any object-type columns that aren't categorical features we want
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    leakage_cols.extend([c for c in object_cols if c not in leakage_cols])
    if leakage_cols:
        X = X.drop(columns=leakage_cols)
        print(f"  Removed {len(leakage_cols)} leakage/object columns")
    
    # Clean column names (remove special characters for LightGBM)
    import re
    clean_cols = {c: re.sub(r'[^\w]', '_', c) for c in X.columns}
    X = X.rename(columns=clean_cols)
    
    # Get class names
    class_names = sorted(df[label_col].unique().tolist())
    
    # Stratified split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=seed  # 0.176 ≈ 15/85
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names


def run_v4_experiment(
    data_path: str,
    output_dir: str = "results_v4",
    seed: int = 42,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Run complete V4 experiment."""
    
    set_global_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    
    print("=" * 70)
    print("V4 EXPERIMENT: THREE-STAGE HIERARCHICAL CLASSIFICATION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_and_prepare_data(data_path, seed)
    
    # Identify categorical features
    cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
    
    # Initialize V4 Pipeline
    print("\n" + "=" * 70)
    print("TRAINING V4 PIPELINE")
    print("=" * 70)
    
    pipeline = V4Pipeline(
        class_names=class_names,
        seed=seed,
        use_gpu=use_gpu,
    )
    
    # Train
    pipeline.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cat_features=cat_features,
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATING V4 ON TEST SET")
    print("=" * 70)
    
    results = evaluate_v4_pipeline(pipeline, X_test, y_test)
    
    # Print results
    print("\n" + "-" * 50)
    print("V4 TEST SET RESULTS")
    print("-" * 50)
    print(f"Accuracy:         {results['accuracy']*100:.2f}%")
    print(f"Macro Precision:  {results['macro_precision']*100:.2f}%")
    print(f"Macro Recall:     {results['macro_recall']*100:.2f}%")
    print(f"Macro F1:         {results['macro_f1']*100:.2f}%")
    print(f"Weighted F1:      {results['weighted_f1']*100:.2f}%")
    print(f"MCC:              {results['mcc']:.4f}")
    
    print("\nPer-Class F1 Scores:")
    for m in results['per_class_metrics']:
        print(f"  {m['class']:20s}: P={m['precision']*100:5.2f}% R={m['recall']*100:5.2f}% F1={m['f1']*100:5.2f}% (n={int(m['support'])})")
    
    if results['stage3_corrections']:
        print(f"\nStage 3 Corrections: {results['stage3_corrections']}")
    
    # Save results
    results_summary = {
        "accuracy": results["accuracy"],
        "macro_precision": results["macro_precision"],
        "macro_recall": results["macro_recall"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "mcc": results["mcc"],
        "stage3_corrections": results["stage3_corrections"],
    }
    write_json(os.path.join(output_dir, "v4_results_summary.json"), results_summary)
    
    # Save per-class metrics
    per_class_df = pd.DataFrame(results['per_class_metrics'])
    per_class_df.to_csv(os.path.join(output_dir, "v4_per_class_metrics.csv"), index=False)
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        results['confusion_matrix'],
        index=class_names,
        columns=class_names
    )
    cm_df.to_csv(os.path.join(output_dir, "v4_confusion_matrix.csv"))
    
    # Save pipeline
    pipeline.save(os.path.join(output_dir, "models", "v4_pipeline.pkl"))
    
    print(f"\nResults saved to {output_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


def compare_v3_v4(v3_results_path: str, v4_results: Dict[str, Any], output_dir: str):
    """Compare V3 and V4 results and generate comparison report."""
    
    # Load V3 results
    with open(v3_results_path, 'r') as f:
        v3_data = json.load(f)
    
    # V3 metrics (from end_to_end_summary.json)
    v3_metrics = {
        "accuracy": v3_data.get("accuracy", 0),
        "macro_f1": v3_data.get("macro_f1", 0),
        "weighted_f1": v3_data.get("weighted_f1", 0),
    }
    
    # V4 metrics
    v4_metrics = {
        "accuracy": v4_results["accuracy"],
        "macro_precision": v4_results["macro_precision"],
        "macro_recall": v4_results["macro_recall"],
        "macro_f1": v4_results["macro_f1"],
        "weighted_f1": v4_results["weighted_f1"],
        "mcc": v4_results["mcc"],
    }
    
    # Calculate improvements
    comparison = {
        "v3": v3_metrics,
        "v4": v4_metrics,
        "improvements": {
            "accuracy": (v4_metrics["accuracy"] - v3_metrics["accuracy"]) * 100,
            "macro_f1": (v4_metrics["macro_f1"] - v3_metrics["macro_f1"]) * 100,
            "weighted_f1": (v4_metrics["weighted_f1"] - v3_metrics["weighted_f1"]) * 100,
        }
    }
    
    write_json(os.path.join(output_dir, "v3_v4_comparison.json"), comparison)
    
    print("\n" + "=" * 70)
    print("V3 vs V4 COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<20} {'V3':>12} {'V4':>12} {'Improvement':>12}")
    print("-" * 56)
    print(f"{'Accuracy':<20} {v3_metrics['accuracy']*100:>11.2f}% {v4_metrics['accuracy']*100:>11.2f}% {comparison['improvements']['accuracy']:>+11.2f}%")
    print(f"{'Macro F1':<20} {v3_metrics['macro_f1']*100:>11.2f}% {v4_metrics['macro_f1']*100:>11.2f}% {comparison['improvements']['macro_f1']:>+11.2f}%")
    print(f"{'Weighted F1':<20} {v3_metrics['weighted_f1']*100:>11.2f}% {v4_metrics['weighted_f1']*100:>11.2f}% {comparison['improvements']['weighted_f1']:>+11.2f}%")
    
    return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run V4 experiment")
    parser.add_argument("--data", type=str, default="data/cicddos2019_dataset.csv", help="Path to dataset")
    parser.add_argument("--output", type=str, default="results_v4", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU")
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_v4_experiment(
        data_path=args.data,
        output_dir=args.output,
        seed=args.seed,
        use_gpu=not args.no_gpu,
    )
    
    # Compare with V3 if results exist
    v3_results_path = "results/end_to_end_summary.json"
    if os.path.exists(v3_results_path):
        compare_v3_v4(v3_results_path, results, args.output)
