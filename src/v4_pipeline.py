"""
V4 Pipeline: Complete Three-Stage Hierarchical Classification System
Integrates Stage 1 (Binary Ensemble), Stage 2 (Stacking), Stage 3 (Verification).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import pickle
import os

from .v4_stage1_ensemble import Stage1Ensemble, train_stage1_ensemble
from .v4_stage2_stacking import Stage2StackingEnsemble, train_stage2_stacking
from .v4_stage3_verification import Stage3Verification, train_stage3_verification


class V4Pipeline:
    """Complete V4 Three-Stage Hierarchical Classification Pipeline."""
    
    def __init__(
        self,
        class_names: List[str],
        seed: int = 42,
        use_gpu: bool = True,
    ):
        self.class_names = class_names
        self.attack_classes = [c for c in class_names if c != "Benign"]
        self.seed = seed
        self.use_gpu = use_gpu
        
        self.stage1: Optional[Stage1Ensemble] = None
        self.stage2: Optional[Stage2StackingEnsemble] = None
        self.stage3: Optional[Stage3Verification] = None
        
        self.stage1_metrics: Dict[str, float] = {}
        self.stage2_metrics: Dict[str, float] = {}
        self.stage3_stats: Dict[str, int] = {}
        
        self.is_fitted = False
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        cat_features: Optional[List[str]] = None,
    ) -> "V4Pipeline":
        """Train all three stages of the pipeline."""
        
        print("=" * 60)
        print("V4 PIPELINE TRAINING")
        print("=" * 60)
        
        # Prepare binary labels for Stage 1
        y_train_binary = (y_train != "Benign").astype(int)
        y_val_binary = (y_val != "Benign").astype(int)
        
        # ==================== STAGE 1 ====================
        print("\n" + "=" * 60)
        print("STAGE 1: Binary Detection Ensemble")
        print("=" * 60)
        
        self.stage1, self.stage1_metrics = train_stage1_ensemble(
            X_train=X_train,
            y_train=y_train_binary,
            X_val=X_val,
            y_val=y_val_binary,
            cat_features=cat_features,
            seed=self.seed,
            use_gpu=self.use_gpu,
        )
        
        print(f"\nStage 1 Validation Metrics:")
        for k, v in self.stage1_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Get Stage 1 confidence for Stage 2
        stage1_train_conf = self.stage1.get_attack_confidence(X_train)
        stage1_val_conf = self.stage1.get_attack_confidence(X_val)
        
        # Filter to attack samples for Stage 2 training
        train_attack_mask = y_train != "Benign"
        val_attack_mask = y_val != "Benign"
        
        X_train_attack = X_train[train_attack_mask]
        y_train_attack = y_train[train_attack_mask]
        stage1_train_conf_attack = stage1_train_conf[train_attack_mask]
        
        X_val_attack = X_val[val_attack_mask]
        y_val_attack = y_val[val_attack_mask]
        stage1_val_conf_attack = stage1_val_conf[val_attack_mask]
        
        # ==================== STAGE 2 ====================
        print("\n" + "=" * 60)
        print("STAGE 2: Multiclass Stacking Ensemble")
        print("=" * 60)
        
        self.stage2, self.stage2_metrics = train_stage2_stacking(
            X_train=X_train_attack,
            y_train=y_train_attack,
            X_val=X_val_attack,
            y_val=y_val_attack,
            class_names=self.attack_classes,
            stage1_train_confidence=stage1_train_conf_attack,
            stage1_val_confidence=stage1_val_conf_attack,
            cat_features=cat_features,
            seed=self.seed,
            use_gpu=self.use_gpu,
        )
        
        print(f"\nStage 2 Validation Metrics:")
        for k, v in self.stage2_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # ==================== STAGE 3 ====================
        print("\n" + "=" * 60)
        print("STAGE 3: Verification & Correction Layer")
        print("=" * 60)
        
        self.stage3 = train_stage3_verification(
            X_train=X_train_attack,
            y_train=y_train_attack,
            X_val=X_val_attack,
            y_val=y_val_attack,
            class_names=self.attack_classes,
            cat_features=cat_features,
            seed=self.seed,
            use_gpu=self.use_gpu,
        )
        
        self.is_fitted = True
        
        print("\n" + "=" * 60)
        print("V4 PIPELINE TRAINING COMPLETE")
        print("=" * 60)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Full end-to-end prediction through all three stages."""
        
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        n_samples = len(X)
        predictions = np.empty(n_samples, dtype=object)
        
        # Stage 1: Binary detection
        stage1_preds = self.stage1.predict(X)
        stage1_conf = self.stage1.get_attack_confidence(X)
        
        # Benign predictions (Stage 1 negative)
        benign_mask = stage1_preds == 0
        predictions[benign_mask] = "Benign"
        
        # Attack samples go through Stage 2 and 3
        attack_mask = stage1_preds == 1
        
        if np.sum(attack_mask) == 0:
            return predictions
        
        X_attack = X[attack_mask]
        stage1_conf_attack = stage1_conf[attack_mask]
        
        # Stage 2: Multiclass classification
        stage2_preds = self.stage2.predict(X_attack, stage1_conf_attack)
        stage2_proba = self.stage2.predict_proba(X_attack, stage1_conf_attack)
        stage2_conf = np.max(stage2_proba, axis=1)
        
        # Stage 3: Verification and correction
        stage3_preds, stage3_conf, self.stage3_stats = self.stage3.verify_and_correct(
            X=X_attack,
            stage2_predictions=stage2_preds,
            stage2_confidence=stage2_conf,
        )
        
        predictions[attack_mask] = stage3_preds
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction probabilities.
        Returns tuple of (attack_probability, class_probabilities).
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Stage 1 probabilities
        stage1_proba = self.stage1.predict_proba(X)
        attack_prob = stage1_proba[:, 1]
        
        # Initialize class probabilities (including Benign)
        n_samples = len(X)
        n_classes = len(self.class_names)
        class_proba = np.zeros((n_samples, n_classes))
        
        # Benign probability from Stage 1
        benign_idx = self.class_names.index("Benign") if "Benign" in self.class_names else 0
        class_proba[:, benign_idx] = stage1_proba[:, 0]
        
        # Attack samples: get Stage 2 probabilities
        attack_mask = attack_prob >= 0.5
        
        if np.sum(attack_mask) > 0:
            X_attack = X[attack_mask]
            stage1_conf_attack = attack_prob[attack_mask]
            
            stage2_proba = self.stage2.predict_proba(X_attack, stage1_conf_attack)
            
            # Map Stage 2 probabilities to full class space
            for i, attack_class in enumerate(self.attack_classes):
                if attack_class in self.class_names:
                    class_idx = self.class_names.index(attack_class)
                    class_proba[attack_mask, class_idx] = stage2_proba[:, i] * attack_prob[attack_mask]
        
        return attack_prob, class_proba
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all stage metrics."""
        return {
            "stage1": self.stage1_metrics,
            "stage2": self.stage2_metrics,
            "stage3_corrections": self.stage3_stats,
        }
    
    def save(self, filepath: str) -> None:
        """Save the entire pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"V4 Pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "V4Pipeline":
        """Load a saved pipeline from disk."""
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)
        print(f"V4 Pipeline loaded from {filepath}")
        return pipeline


def evaluate_v4_pipeline(
    pipeline: V4Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Comprehensive evaluation of V4 pipeline."""
    
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        f1_score,
        matthews_corrcoef,
        classification_report,
        confusion_matrix,
    )
    
    y_pred = pipeline.predict(X_test)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Per-class metrics
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=pipeline.class_names, zero_division=0
    )
    
    per_class_metrics = []
    for i, class_name in enumerate(pipeline.class_names):
        per_class_metrics.append({
            "class": class_name,
            "precision": class_precision[i],
            "recall": class_recall[i],
            "f1": class_f1[i],
            "support": class_support[i],
        })
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.class_names)
    
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm,
        "class_names": pipeline.class_names,
        "stage3_corrections": pipeline.stage3_stats,
    }
