"""Predictor Package for StockAI.

Provides ML-based prediction capabilities:
- Feature engineering pipeline
- XGBoost classifier
- LSTM sequence model
- Ensemble predictor
- Prediction accuracy tracking
"""

from stockai.core.predictor.accuracy import PredictionAccuracyTracker
from stockai.core.predictor.features import FeatureEngineer, generate_features
from stockai.core.predictor.xgboost_model import XGBoostPredictor
from stockai.core.predictor.lstm_model import LSTMPredictor
from stockai.core.predictor.ensemble import EnsemblePredictor

__all__ = [
    "FeatureEngineer",
    "generate_features",
    "XGBoostPredictor",
    "LSTMPredictor",
    "EnsemblePredictor",
    "PredictionAccuracyTracker",
]
