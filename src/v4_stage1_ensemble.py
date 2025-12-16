"""
V4 Stage 1: Binary Detection Ensemble
Three-model soft voting ensemble for Benign vs Attack classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class Stage1Ensemble:
    """Ensemble of LightGBM, XGBoost, and CatBoost for binary classification."""
    
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        seed: int = 42,
        use_gpu: bool = True,
    ):
        self.weights = weights if weights else [0.35, 0.30, 0.35]  # lgbm, xgb, catboost
        self.seed = seed
        self.use_gpu = use_gpu
        self.models = {}
        self.is_fitted = False
        
    def _get_lgbm_params(self) -> Dict[str, Any]:
        return {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": self.seed,
            "verbose": -1,
            "n_jobs": -1,
        }
    
    def _get_xgb_params(self) -> Dict[str, Any]:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.seed,
            "verbosity": 0,
            "n_jobs": -1,
        }
        if self.use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
        return params
    
    def _get_catboost_params(self) -> Dict[str, Any]:
        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "auto_class_weights": "Balanced",
            "random_seed": self.seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        if self.use_gpu:
            params["task_type"] = "GPU"
            params["devices"] = "0"
        return params
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[List[str]] = None,
        early_stopping_rounds: int = 50,
    ) -> "Stage1Ensemble":
        """Fit all three models in the ensemble."""
        
        print("Training Stage 1 Ensemble...")
        
        # 1. Train LightGBM
        print("  Training LightGBM...")
        lgbm_params = self._get_lgbm_params()
        self.models["lgbm"] = lgb.LGBMClassifier(**lgbm_params)
        
        if X_val is not None:
            self.models["lgbm"].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
        else:
            self.models["lgbm"].fit(X_train, y_train)
        
        # 2. Train XGBoost
        print("  Training XGBoost...")
        xgb_params = self._get_xgb_params()
        
        # Calculate scale_pos_weight for imbalanced data
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        xgb_params["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0
        
        self.models["xgb"] = xgb.XGBClassifier(**xgb_params)
        
        if X_val is not None:
            self.models["xgb"].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.models["xgb"].fit(X_train, y_train)
        
        # 3. Train CatBoost
        if CATBOOST_AVAILABLE:
            print("  Training CatBoost...")
            cat_params = self._get_catboost_params()
            self.models["catboost"] = CatBoostClassifier(**cat_params)
            
            cat_indices = [i for i, col in enumerate(X_train.columns) if col in (cat_features or [])]
            
            if X_val is not None:
                self.models["catboost"].fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    cat_features=cat_indices if cat_indices else None,
                    early_stopping_rounds=early_stopping_rounds,
                )
            else:
                self.models["catboost"].fit(
                    X_train, y_train,
                    cat_features=cat_indices if cat_indices else None,
                )
        else:
            print("  CatBoost not available, using 2-model ensemble")
            self.weights = [0.5, 0.5, 0.0]
        
        self.is_fitted = True
        print("Stage 1 Ensemble training complete.")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get soft voting probabilities from ensemble."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        proba_lgbm = self.models["lgbm"].predict_proba(X)
        proba_xgb = self.models["xgb"].predict_proba(X)
        
        if CATBOOST_AVAILABLE and "catboost" in self.models:
            proba_cat = self.models["catboost"].predict_proba(X)
            ensemble_proba = (
                self.weights[0] * proba_lgbm +
                self.weights[1] * proba_xgb +
                self.weights[2] * proba_cat
            )
        else:
            ensemble_proba = (
                self.weights[0] * proba_lgbm +
                self.weights[1] * proba_xgb
            ) / (self.weights[0] + self.weights[1])
        
        return ensemble_proba
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def get_attack_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """Get attack probability scores for confidence propagation to Stage 2."""
        proba = self.predict_proba(X)
        return proba[:, 1]


def train_stage1_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_features: Optional[List[str]] = None,
    seed: int = 42,
    use_gpu: bool = True,
) -> Tuple[Stage1Ensemble, Dict[str, float]]:
    """Train Stage 1 ensemble and return model with metrics."""
    
    ensemble = Stage1Ensemble(seed=seed, use_gpu=use_gpu)
    ensemble.fit(X_train, y_train, X_val, y_val, cat_features)
    
    # Evaluate on validation set
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
    
    y_pred = ensemble.predict(X_val)
    y_proba = ensemble.predict_proba(X_val)[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", pos_label=1)
    
    metrics = {
        "roc_auc": roc_auc_score(y_val, y_proba),
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    return ensemble, metrics
