"""
V4 Stage 3: Verification & Correction Layer
Confusion-aware specialist models for targeted error correction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


# Define confusion-prone class groups based on error analysis
CONFUSION_GROUPS = {
    "udp_group": ["UDP-lag", "UDP_Attack"],
    "dns_group": ["DrDoS_DNS", "LDAP_Attack", "DrDoS_SNMP"],
    "minority_group": ["Portmap", "NetBIOS_Attack"],
}


class SpecialistModel:
    """Binary or multi-class specialist for a specific confusion group."""
    
    def __init__(
        self,
        name: str,
        target_classes: List[str],
        seed: int = 42,
        use_gpu: bool = True,
    ):
        self.name = name
        self.target_classes = target_classes
        self.n_classes = len(target_classes)
        self.seed = seed
        self.use_gpu = use_gpu
        self.model = None
        self.label_encoder = None
        self.is_fitted = False
        
    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import LabelEncoder
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.target_classes)
        return self.label_encoder.transform(y)
    
    def _decode_labels(self, y: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y)
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[List[str]] = None,
    ) -> "SpecialistModel":
        """Train specialist model on subset of data."""
        
        # Filter to only target classes
        mask_train = np.isin(y_train, self.target_classes)
        X_train_sub = X_train[mask_train]
        y_train_sub = y_train[mask_train]
        
        if len(X_train_sub) == 0:
            print(f"  Warning: No samples for specialist {self.name}")
            return self
        
        y_train_encoded = self._encode_labels(y_train_sub)
        
        if X_val is not None and y_val is not None:
            mask_val = np.isin(y_val, self.target_classes)
            X_val_sub = X_val[mask_val]
            y_val_sub = y_val[mask_val]
            y_val_encoded = self._encode_labels(y_val_sub) if len(y_val_sub) > 0 else None
        else:
            X_val_sub, y_val_encoded = None, None
        
        # Use XGBoost for specialists (good performance, fast training)
        if self.n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "auc"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
        
        params = {
            "objective": objective,
            "eval_metric": eval_metric,
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.seed,
            "verbosity": 0,
            "n_jobs": -1,
        }
        
        if self.n_classes > 2:
            params["num_class"] = self.n_classes
        
        if self.use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
        
        self.model = xgb.XGBClassifier(**params)
        
        if X_val_sub is not None and len(X_val_sub) > 0:
            self.model.fit(
                X_train_sub, y_train_encoded,
                eval_set=[(X_val_sub, y_val_encoded)],
                verbose=False
            )
        else:
            self.model.fit(X_train_sub, y_train_encoded)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get class probabilities."""
        if not self.is_fitted:
            return None
        return self.model.predict_proba(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            return None
        proba = self.predict_proba(X)
        if self.n_classes == 2:
            y_pred_encoded = (proba[:, 1] >= 0.5).astype(int)
        else:
            y_pred_encoded = np.argmax(proba, axis=1)
        return self._decode_labels(y_pred_encoded)
    
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """Get max probability as confidence score."""
        proba = self.predict_proba(X)
        if proba is None:
            return np.zeros(len(X))
        return np.max(proba, axis=1)


class Stage3Verification:
    """Verification layer with confusion-aware specialist models."""
    
    def __init__(
        self,
        class_names: List[str],
        confusion_threshold: float = 0.7,
        specialist_threshold: float = 0.75,
        seed: int = 42,
        use_gpu: bool = True,
    ):
        self.class_names = class_names
        self.confusion_threshold = confusion_threshold
        self.specialist_threshold = specialist_threshold
        self.seed = seed
        self.use_gpu = use_gpu
        self.specialists = {}
        self.confusion_prone_classes = set()
        self.is_fitted = False
        
        # Identify confusion-prone classes
        for group_name, classes in CONFUSION_GROUPS.items():
            for cls in classes:
                if cls in class_names:
                    self.confusion_prone_classes.add(cls)
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        cat_features: Optional[List[str]] = None,
    ) -> "Stage3Verification":
        """Train all specialist models."""
        
        print("Training Stage 3 Verification Specialists...")
        
        for group_name, target_classes in CONFUSION_GROUPS.items():
            # Filter to classes that exist in our data
            valid_classes = [c for c in target_classes if c in self.class_names]
            
            if len(valid_classes) < 2:
                continue
            
            print(f"  Training {group_name} specialist for {valid_classes}...")
            
            specialist = SpecialistModel(
                name=group_name,
                target_classes=valid_classes,
                seed=self.seed,
                use_gpu=self.use_gpu,
            )
            specialist.fit(X_train, y_train, X_val, y_val, cat_features)
            
            if specialist.is_fitted:
                self.specialists[group_name] = specialist
        
        self.is_fitted = True
        print("Stage 3 Verification training complete.")
        return self
    
    def _get_specialist_for_class(self, class_name: str) -> Optional[SpecialistModel]:
        """Find the specialist responsible for a given class."""
        for group_name, classes in CONFUSION_GROUPS.items():
            if class_name in classes and group_name in self.specialists:
                return self.specialists[group_name]
        return None
    
    def verify_and_correct(
        self,
        X: pd.DataFrame,
        stage2_predictions: np.ndarray,
        stage2_confidence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Verify Stage 2 predictions and correct if specialists are more confident.
        
        Returns:
            - corrected_predictions: Final predictions after verification
            - final_confidence: Confidence scores for final predictions
            - correction_stats: Statistics on corrections made
        """
        
        if not self.is_fitted:
            return stage2_predictions, stage2_confidence, {}
        
        corrected_predictions = stage2_predictions.copy()
        final_confidence = stage2_confidence.copy()
        
        correction_stats = {
            "total_reviewed": 0,
            "total_corrected": 0,
        }
        for group_name in self.specialists:
            correction_stats[f"{group_name}_corrected"] = 0
        
        # Process each sample
        for i in range(len(X)):
            pred_class = stage2_predictions[i]
            pred_conf = stage2_confidence[i]
            
            # Check if this prediction is in a confusion-prone group
            if pred_class not in self.confusion_prone_classes:
                continue
            
            # Low confidence predictions need verification
            if pred_conf > self.confusion_threshold:
                continue
            
            correction_stats["total_reviewed"] += 1
            
            # Get the relevant specialist
            specialist = self._get_specialist_for_class(pred_class)
            if specialist is None:
                continue
            
            # Get specialist prediction
            X_sample = X.iloc[[i]]
            specialist_pred = specialist.predict(X_sample)
            specialist_conf = specialist.get_confidence(X_sample)
            
            if specialist_pred is None:
                continue
            
            specialist_pred = specialist_pred[0]
            specialist_conf = specialist_conf[0]
            
            # Override if specialist is more confident
            if specialist_conf > self.specialist_threshold and specialist_conf > pred_conf:
                corrected_predictions[i] = specialist_pred
                final_confidence[i] = specialist_conf
                correction_stats["total_corrected"] += 1
                correction_stats[f"{specialist.name}_corrected"] += 1
        
        return corrected_predictions, final_confidence, correction_stats


def train_stage3_verification(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    class_names: List[str],
    cat_features: Optional[List[str]] = None,
    seed: int = 42,
    use_gpu: bool = True,
) -> Stage3Verification:
    """Train Stage 3 verification layer."""
    
    verifier = Stage3Verification(
        class_names=class_names,
        seed=seed,
        use_gpu=use_gpu,
    )
    
    verifier.fit(X_train, y_train, X_val, y_val, cat_features)
    
    return verifier
