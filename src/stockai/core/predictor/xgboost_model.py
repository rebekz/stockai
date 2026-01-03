"""XGBoost Classifier for StockAI Direction Prediction.

Provides fast baseline predictions with interpretable feature importance.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from stockai.core.predictor.features import FeatureEngineer, create_target, ALL_FEATURES

logger = logging.getLogger(__name__)


class XGBoostPredictor:
    """XGBoost-based direction classifier.

    Features:
    - Binary classification: UP (1) vs DOWN (0)
    - Walk-forward cross-validation
    - Feature importance extraction
    - Fast inference (<100ms)
    - Model serialization to JSON
    """

    DEFAULT_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        model_params: dict[str, Any] | None = None,
        model_path: Path | None = None,
    ):
        """Initialize XGBoost predictor.

        Args:
            model_params: XGBoost hyperparameters
            model_path: Path to load/save model
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        self.params = {**self.DEFAULT_PARAMS, **(model_params or {})}
        self.model_path = model_path or Path("data/models/xgboost_v1.json")
        self.model: xgb.XGBClassifier | None = None
        self.feature_engineer = FeatureEngineer(normalize=True)
        self.feature_names: list[str] = []
        self.training_metrics: dict[str, Any] = {}

    def train(
        self,
        train_df: pd.DataFrame,
        horizon: int = 3,
        validation_split: float = 0.2,
    ) -> dict[str, float]:
        """Train the XGBoost model.

        Args:
            train_df: OHLCV DataFrame (requires 3+ years of data)
            horizon: Days ahead for prediction target
            validation_split: Fraction for validation

        Returns:
            Training metrics (accuracy, auc, etc.)
        """
        logger.info(f"Training XGBoost model with {len(train_df)} samples")

        # Generate features
        features = self.feature_engineer.generate_features(train_df)
        self.feature_names = list(features.columns)

        # Create target
        target = create_target(train_df.loc[features.index], horizon=horizon)

        # Align features and target
        valid_idx = target.dropna().index.intersection(features.index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]

        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples")

        # Time-based split (use last portion for validation)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train model
        self.model = xgb.XGBClassifier(**self.params)

        eval_set = [(X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Calculate metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        train_prob = self.model.predict_proba(X_train)[:, 1]
        val_prob = self.model.predict_proba(X_val)[:, 1]

        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

        self.training_metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "val_accuracy": float(accuracy_score(y_val, val_pred)),
            "train_auc": float(roc_auc_score(y_train, train_prob)),
            "val_auc": float(roc_auc_score(y_val, val_prob)),
            "val_precision": float(precision_score(y_val, val_pred)),
            "val_recall": float(recall_score(y_val, val_pred)),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": len(self.feature_names),
        }

        logger.info(f"Training complete. Val accuracy: {self.training_metrics['val_accuracy']:.4f}")
        return self.training_metrics

    def train_walkforward(
        self,
        df: pd.DataFrame,
        horizon: int = 3,
        n_splits: int = 4,
        test_size_months: int = 3,
    ) -> dict[str, Any]:
        """Train with walk-forward cross-validation.

        Args:
            df: Full OHLCV DataFrame
            horizon: Days ahead for prediction
            n_splits: Number of walk-forward splits
            test_size_months: Test window size in months

        Returns:
            Cross-validation metrics
        """
        logger.info(f"Walk-forward training with {n_splits} splits")

        features = self.feature_engineer.generate_features(df)
        target = create_target(df.loc[features.index], horizon=horizon)

        valid_idx = target.dropna().index.intersection(features.index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]

        # Calculate split points
        test_samples = test_size_months * 20  # ~20 trading days per month
        min_train = len(X) - (n_splits * test_samples)

        if min_train < 100:
            raise ValueError(f"Insufficient data for walk-forward: {len(X)} samples")

        results = []
        best_model = None
        best_auc = 0.0

        for i in range(n_splits):
            test_end = len(X) - (i * test_samples)
            test_start = test_end - test_samples
            train_end = test_start

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            # Train model for this fold
            model = xgb.XGBClassifier(**self.params)
            model.fit(X_train, y_train, verbose=False)

            # Evaluate
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

            from sklearn.metrics import accuracy_score, roc_auc_score

            accuracy = float(accuracy_score(y_test, pred))
            auc = float(roc_auc_score(y_test, prob))

            results.append({
                "fold": i + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": accuracy,
                "auc": auc,
            })

            if auc > best_auc:
                best_auc = auc
                best_model = model

            logger.info(f"Fold {i+1}: accuracy={accuracy:.4f}, auc={auc:.4f}")

        # Use best model
        self.model = best_model
        self.feature_names = list(X.columns)

        # Aggregate metrics
        cv_metrics = {
            "mean_accuracy": np.mean([r["accuracy"] for r in results]),
            "std_accuracy": np.std([r["accuracy"] for r in results]),
            "mean_auc": np.mean([r["auc"] for r in results]),
            "std_auc": np.std([r["auc"] for r in results]),
            "folds": results,
        }

        self.training_metrics = cv_metrics
        return cv_metrics

    def predict(
        self,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Predict direction for given data.

        Args:
            df: OHLCV DataFrame (at least 50 rows for features)

        Returns:
            Prediction with probability and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        start_time = time.time()

        # Generate features
        features = self.feature_engineer.generate_features(df)

        if features.empty:
            raise ValueError("Could not generate features from data")

        # Use last row for prediction
        X = features.iloc[[-1]]

        # Ensure feature alignment
        for feat in self.feature_names:
            if feat not in X.columns:
                X[feat] = 0.0
        X = X[self.feature_names]

        # Predict
        prob = float(self.model.predict_proba(X)[0, 1])
        direction = "UP" if prob > 0.5 else "DOWN"
        confidence = abs(prob - 0.5) * 2  # Scale to 0-1

        inference_time = (time.time() - start_time) * 1000  # ms

        return {
            "direction": direction,
            "probability": prob,
            "confidence": confidence,
            "model": "xgboost",
            "inference_time_ms": inference_time,
        }

    def predict_proba(self, df: pd.DataFrame) -> float:
        """Get prediction probability (for ensemble).

        Args:
            df: OHLCV DataFrame

        Returns:
            Probability of UP direction (0-1)
        """
        result = self.predict(df)
        return result["probability"]

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained")

        importance = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importance))

        return self.feature_engineer.get_feature_importance(importance_dict)

    def save(self, path: Path | None = None) -> bool:
        """Save model to JSON file.

        Args:
            path: Save path (defaults to model_path)

        Returns:
            True if successful
        """
        if self.model is None:
            raise ValueError("No model to save")

        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save model
            self.model.save_model(str(save_path))

            # Save metadata
            meta_path = save_path.with_suffix(".meta.json")
            metadata = {
                "feature_names": self.feature_names,
                "training_metrics": self.training_metrics,
                "params": self.params,
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load(self, path: Path | None = None) -> bool:
        """Load model from JSON file.

        Args:
            path: Load path (defaults to model_path)

        Returns:
            True if successful
        """
        load_path = path or self.model_path

        if not load_path.exists():
            logger.warning(f"Model file not found: {load_path}")
            return False

        try:
            # Load model
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(load_path))

            # Load metadata
            meta_path = load_path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
                self.feature_names = metadata.get("feature_names", [])
                self.training_metrics = metadata.get("training_metrics", {})
                self.params = metadata.get("params", self.params)

            logger.info(f"Model loaded from {load_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
