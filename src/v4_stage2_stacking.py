"""
V4 Stage 2: Multiclass Stacking Ensemble
Three-model stacking with meta-learner for attack type classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class Stage2StackingEnsemble:
    """Stacking ensemble with meta-learner for multiclass attack classification."""
    
    def __init__(
        self,
        n_classes: int,
        class_names: List[str],
        seed: int = 42,
        use_gpu: bool = True,
        n_folds: int = 5,
    ):
        self.n_classes = n_classes
        self.class_names = class_names
        self.seed = seed
        self.use_gpu = use_gpu
        self.n_folds = n_folds
        self.base_models = {}
        self.meta_learner = None
        self.label_encoder = None
        self.is_fitted = False
        
    def _get_catboost_params(self) -> Dict[str, Any]:
        params = {
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1:average=Macro",
            "iterations": 5000,
            "learning_rate": 0.02,
            "depth": 10,
            "l2_leaf_reg": 0.5,
            "random_strength": 0.15,
            "bagging_temperature": 0.1,
            "auto_class_weights": "Balanced",
            "border_count": 200,
            "min_data_in_leaf": 5,
            "grow_policy": "Lossguide",
            "max_leaves": 64,
            "random_seed": self.seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        if self.use_gpu:
            params["task_type"] = "GPU"
            params["devices"] = "0"
        return params
    
    def _get_lgbm_params(self) -> Dict[str, Any]:
        return {
            "objective": "multiclass",
            "num_class": self.n_classes,
            "metric": "multi_logloss",
            "n_estimators": 3000,
            "learning_rate": 0.02,
            "num_leaves": 64,
            "max_depth": 10,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": self.seed,
            "verbose": -1,
            "n_jobs": -1,
        }
    
    def _get_xgb_params(self) -> Dict[str, Any]:
        params = {
            "objective": "multi:softprob",
            "num_class": self.n_classes,
            "eval_metric": "mlogloss",
            "n_estimators": 3000,
            "learning_rate": 0.02,
            "max_depth": 10,
            "min_child_weight": 5,
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
    
    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Encode string labels to integers."""
        from sklearn.preprocessing import LabelEncoder
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.class_names)
        return self.label_encoder.transform(y)
    
    def _decode_labels(self, y: np.ndarray) -> np.ndarray:
        """Decode integer labels back to strings."""
        return self.label_encoder.inverse_transform(y)
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        stage1_confidence: Optional[np.ndarray] = None,
        stage1_val_confidence: Optional[np.ndarray] = None,
        cat_features: Optional[List[str]] = None,
        early_stopping_rounds: int = 100,
    ) -> "Stage2StackingEnsemble":
        """Fit stacking ensemble with out-of-fold predictions for meta-learner."""
        
        print("Training Stage 2 Stacking Ensemble...")
        
        # Encode labels
        y_train_encoded = self._encode_labels(y_train)
        if y_val is not None:
            y_val_encoded = self._encode_labels(y_val)
        
        # Add Stage 1 confidence as feature if provided
        if stage1_confidence is not None:
            X_train = X_train.copy()
            X_train["stage1_attack_confidence"] = stage1_confidence
            if X_val is not None and stage1_val_confidence is not None:
                X_val = X_val.copy()
                X_val["stage1_attack_confidence"] = stage1_val_confidence
        
        # Get out-of-fold predictions for meta-learner training
        n_samples = len(X_train)
        oof_preds_cat = np.zeros((n_samples, self.n_classes))
        oof_preds_lgbm = np.zeros((n_samples, self.n_classes))
        oof_preds_xgb = np.zeros((n_samples, self.n_classes))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        cat_indices = [i for i, col in enumerate(X_train.columns) if col in (cat_features or [])]
        
        print(f"  Generating OOF predictions with {self.n_folds}-fold CV...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
            print(f"    Fold {fold + 1}/{self.n_folds}...")
            
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train_encoded[train_idx], y_train_encoded[val_idx]
            
            # CatBoost
            if CATBOOST_AVAILABLE:
                cat_model = CatBoostClassifier(**self._get_catboost_params())
                cat_model.fit(
                    X_tr, y_tr,
                    eval_set=(X_vl, y_vl),
                    cat_features=cat_indices if cat_indices else None,
                    early_stopping_rounds=early_stopping_rounds,
                )
                oof_preds_cat[val_idx] = cat_model.predict_proba(X_vl)
            
            # LightGBM
            lgbm_model = lgb.LGBMClassifier(**self._get_lgbm_params())
            lgbm_model.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
            oof_preds_lgbm[val_idx] = lgbm_model.predict_proba(X_vl)
            
            # XGBoost
            xgb_model = xgb.XGBClassifier(**self._get_xgb_params())
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_vl)
        
        # Train final base models on full training data
        print("  Training final base models on full data...")
        
        if CATBOOST_AVAILABLE:
            self.base_models["catboost"] = CatBoostClassifier(**self._get_catboost_params())
            if X_val is not None:
                self.base_models["catboost"].fit(
                    X_train, y_train_encoded,
                    eval_set=(X_val, y_val_encoded),
                    cat_features=cat_indices if cat_indices else None,
                    early_stopping_rounds=early_stopping_rounds,
                )
            else:
                self.base_models["catboost"].fit(
                    X_train, y_train_encoded,
                    cat_features=cat_indices if cat_indices else None,
                )
        
        self.base_models["lgbm"] = lgb.LGBMClassifier(**self._get_lgbm_params())
        if X_val is not None:
            self.base_models["lgbm"].fit(
                X_train, y_train_encoded,
                eval_set=[(X_val, y_val_encoded)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
        else:
            self.base_models["lgbm"].fit(X_train, y_train_encoded)
        
        self.base_models["xgb"] = xgb.XGBClassifier(**self._get_xgb_params())
        if X_val is not None:
            self.base_models["xgb"].fit(
                X_train, y_train_encoded,
                eval_set=[(X_val, y_val_encoded)],
                verbose=False
            )
        else:
            self.base_models["xgb"].fit(X_train, y_train_encoded)
        
        # Train meta-learner on OOF predictions
        print("  Training meta-learner...")
        meta_features = np.hstack([oof_preds_cat, oof_preds_lgbm, oof_preds_xgb])
        
        self.meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=self.seed,
            n_jobs=-1,
        )
        self.meta_learner.fit(meta_features, y_train_encoded)
        
        self.is_fitted = True
        print("Stage 2 Stacking Ensemble training complete.")
        return self
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        stage1_confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get stacked probabilities from ensemble."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Add Stage 1 confidence if provided
        if stage1_confidence is not None:
            X = X.copy()
            X["stage1_attack_confidence"] = stage1_confidence
        
        # Get base model predictions
        pred_cat = self.base_models["catboost"].predict_proba(X) if "catboost" in self.base_models else np.zeros((len(X), self.n_classes))
        pred_lgbm = self.base_models["lgbm"].predict_proba(X)
        pred_xgb = self.base_models["xgb"].predict_proba(X)
        
        # Stack predictions
        meta_features = np.hstack([pred_cat, pred_lgbm, pred_xgb])
        
        # Meta-learner prediction
        return self.meta_learner.predict_proba(meta_features)
    
    def predict(
        self,
        X: pd.DataFrame,
        stage1_confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict class labels (as strings)."""
        proba = self.predict_proba(X, stage1_confidence)
        y_pred_encoded = np.argmax(proba, axis=1)
        return self._decode_labels(y_pred_encoded)
    
    def get_base_predictions(
        self,
        X: pd.DataFrame,
        stage1_confidence: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Get individual base model predictions for analysis."""
        if stage1_confidence is not None:
            X = X.copy()
            X["stage1_attack_confidence"] = stage1_confidence
            
        return {
            "catboost": self.base_models["catboost"].predict_proba(X) if "catboost" in self.base_models else None,
            "lgbm": self.base_models["lgbm"].predict_proba(X),
            "xgb": self.base_models["xgb"].predict_proba(X),
        }


def train_stage2_stacking(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    class_names: List[str],
    stage1_train_confidence: Optional[np.ndarray] = None,
    stage1_val_confidence: Optional[np.ndarray] = None,
    cat_features: Optional[List[str]] = None,
    seed: int = 42,
    use_gpu: bool = True,
) -> Tuple[Stage2StackingEnsemble, Dict[str, float]]:
    """Train Stage 2 stacking ensemble and return model with metrics."""
    
    n_classes = len(class_names)
    
    ensemble = Stage2StackingEnsemble(
        n_classes=n_classes,
        class_names=class_names,
        seed=seed,
        use_gpu=use_gpu,
    )
    
    ensemble.fit(
        X_train, y_train, X_val, y_val,
        stage1_confidence=stage1_train_confidence,
        stage1_val_confidence=stage1_val_confidence,
        cat_features=cat_features,
    )
    
    # Evaluate on validation set
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, f1_score
    )
    
    y_pred = ensemble.predict(X_val, stage1_val_confidence)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0
    )
    
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }
    
    return ensemble, metrics
