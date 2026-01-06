"""Ensemble Predictor for StockAI.

Combines XGBoost and LSTM predictions with sentiment modifier
for robust direction predictions.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from stockai.core.predictor.xgboost_model import XGBoostPredictor
from stockai.core.predictor.lstm_model import LSTMPredictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble predictor combining multiple models.

    Weights:
    - XGBoost: 0.4 (fast, interpretable)
    - LSTM: 0.4 (captures sequences)
    - Sentiment: 0.2 (modifier)

    Features:
    - Weighted probability combination
    - Majority vote with confidence threshold
    - Calibrated confidence output
    - Model contribution visibility
    """

    DEFAULT_WEIGHTS = {
        "xgboost": 0.4,
        "lstm": 0.4,
        "sentiment": 0.2,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        xgboost_path: Path | None = None,
        lstm_path: Path | None = None,
        confidence_threshold: float = 0.55,
    ):
        """Initialize ensemble predictor.

        Args:
            weights: Model weights (must sum to 1.0)
            xgboost_path: Path to XGBoost model
            lstm_path: Path to LSTM model
            confidence_threshold: Minimum for strong prediction
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.confidence_threshold = confidence_threshold

        # Validate weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing")
            for k in self.weights:
                self.weights[k] /= total

        # Initialize models
        self.xgboost = XGBoostPredictor(model_path=xgboost_path)
        self.lstm = LSTMPredictor(model_path=lstm_path)

        self.models_loaded = {"xgboost": False, "lstm": False}

    def load_models(self) -> dict[str, bool]:
        """Load all component models.

        Returns:
            Dict of model name -> load success
        """
        self.models_loaded["xgboost"] = self.xgboost.load()
        self.models_loaded["lstm"] = self.lstm.load()

        loaded_count = sum(self.models_loaded.values())
        logger.info(f"Loaded {loaded_count}/2 models")

        return self.models_loaded

    def predict(
        self,
        df: pd.DataFrame,
        sentiment_score: float | None = None,
    ) -> dict[str, Any]:
        """Generate ensemble prediction.

        Args:
            df: OHLCV DataFrame
            sentiment_score: Optional sentiment (-1 to 1)

        Returns:
            Ensemble prediction with model contributions
        """
        contributions = {}
        weighted_prob = 0.0
        active_weight = 0.0

        # XGBoost prediction
        if self.models_loaded.get("xgboost"):
            try:
                xgb_result = self.xgboost.predict(df)
                xgb_prob = xgb_result["probability"]
                contributions["xgboost"] = {
                    "probability": xgb_prob,
                    "direction": xgb_result["direction"],
                    "weight": self.weights["xgboost"],
                }
                weighted_prob += xgb_prob * self.weights["xgboost"]
                active_weight += self.weights["xgboost"]
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
                contributions["xgboost"] = {"error": str(e)}

        # LSTM prediction
        if self.models_loaded.get("lstm"):
            try:
                lstm_result = self.lstm.predict(df)
                lstm_prob = lstm_result["probability"]
                contributions["lstm"] = {
                    "probability": lstm_prob,
                    "direction": lstm_result["direction"],
                    "weight": self.weights["lstm"],
                }
                weighted_prob += lstm_prob * self.weights["lstm"]
                active_weight += self.weights["lstm"]
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
                contributions["lstm"] = {"error": str(e)}

        # Sentiment modifier
        if sentiment_score is not None:
            # Convert sentiment (-1 to 1) to probability modifier
            # Positive sentiment -> push toward UP
            sentiment_prob = (sentiment_score + 1) / 2  # 0 to 1
            contributions["sentiment"] = {
                "score": sentiment_score,
                "probability": sentiment_prob,
                "weight": self.weights["sentiment"],
            }
            weighted_prob += sentiment_prob * self.weights["sentiment"]
            active_weight += self.weights["sentiment"]
        else:
            # Without sentiment, re-weight other models
            contributions["sentiment"] = {"score": None, "weight": 0}

        # Normalize if not all models contributed
        if active_weight > 0 and active_weight < 1.0:
            weighted_prob /= active_weight

        # Determine direction and confidence
        direction = "UP" if weighted_prob > 0.5 else "DOWN"
        raw_confidence = abs(weighted_prob - 0.5) * 2

        # Calibrate confidence
        calibrated_confidence = self._calibrate_confidence(
            raw_confidence,
            contributions,
        )

        # Determine confidence level
        if calibrated_confidence >= 0.7:
            confidence_level = "HIGH"
        elif calibrated_confidence >= 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Check model agreement
        directions = []
        for model in ["xgboost", "lstm"]:
            if model in contributions and "direction" in contributions[model]:
                directions.append(contributions[model]["direction"])

        agreement = len(set(directions)) == 1 if directions else False

        return {
            "direction": direction,
            "probability": weighted_prob,
            "confidence": calibrated_confidence,
            "confidence_level": confidence_level,
            "model": "ensemble",
            "model_agreement": agreement,
            "contributions": contributions,
            "active_models": sum(1 for m in ["xgboost", "lstm"] if self.models_loaded.get(m)),
        }

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        contributions: dict,
    ) -> float:
        """Calibrate confidence based on model agreement.

        If models disagree, reduce confidence.

        Args:
            raw_confidence: Raw confidence from weighted average
            contributions: Model contributions

        Returns:
            Calibrated confidence (0-1)
        """
        directions = []
        for model in ["xgboost", "lstm"]:
            if model in contributions and "direction" in contributions[model]:
                directions.append(contributions[model]["direction"])

        # If models disagree, reduce confidence
        if len(set(directions)) > 1:
            # Disagreement penalty
            raw_confidence *= 0.7

        # Slight reduction if sentiment not available
        if contributions.get("sentiment", {}).get("score") is None:
            raw_confidence *= 0.95

        return min(raw_confidence, 1.0)

    def predict_with_history(
        self,
        df: pd.DataFrame,
        historical_accuracy: float | None = None,
        sentiment_score: float | None = None,
    ) -> dict[str, Any]:
        """Predict with historical accuracy context.

        Args:
            df: OHLCV DataFrame
            historical_accuracy: Model's historical accuracy for this stock
            sentiment_score: Optional sentiment score

        Returns:
            Prediction with accuracy context
        """
        result = self.predict(df, sentiment_score)

        if historical_accuracy is not None:
            result["historical_accuracy"] = historical_accuracy
            # Adjust confidence based on historical performance
            if historical_accuracy < 0.5:
                result["confidence"] *= 0.8
                result["warning"] = "Model has low historical accuracy for this stock"
            elif historical_accuracy > 0.6:
                result["confidence"] = min(result["confidence"] * 1.1, 1.0)

        return result

    def get_model_summary(self) -> dict[str, Any]:
        """Get summary of loaded models and their status.

        Returns:
            Model status summary
        """
        summary = {
            "models_loaded": self.models_loaded,
            "weights": self.weights,
            "xgboost_metrics": {},
            "lstm_metrics": {},
        }

        if self.models_loaded.get("xgboost"):
            summary["xgboost_metrics"] = self.xgboost.training_metrics

        if self.models_loaded.get("lstm"):
            summary["lstm_metrics"] = self.lstm.training_metrics

        return summary

    def train_all(
        self,
        train_df: pd.DataFrame,
        horizon: int = 3,
        xgboost_params: dict | None = None,
        lstm_params: dict | None = None,
    ) -> dict[str, Any]:
        """Train all component models.

        Args:
            train_df: OHLCV training data
            horizon: Prediction horizon in days
            xgboost_params: XGBoost training params
            lstm_params: LSTM training params

        Returns:
            Training metrics for all models
        """
        results = {}

        # Train XGBoost
        logger.info("Training XGBoost model...")
        try:
            # Reinitialize with custom params if provided
            if xgboost_params:
                self.xgboost = XGBoostPredictor(
                    model_params=xgboost_params,
                    model_path=self.xgboost.model_path,
                )
            xgb_metrics = self.xgboost.train(
                train_df,
                horizon=horizon,
            )
            results["xgboost"] = xgb_metrics
            self.models_loaded["xgboost"] = True
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            results["xgboost"] = {"error": str(e)}

        # Train LSTM
        logger.info("Training LSTM model...")
        try:
            lstm_metrics = self.lstm.train(
                train_df,
                horizon=horizon,
                **(lstm_params or {}),
            )
            results["lstm"] = lstm_metrics
            self.models_loaded["lstm"] = True
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            results["lstm"] = {"error": str(e)}

        return results

    def save_all(self) -> dict[str, bool]:
        """Save all trained models.

        Returns:
            Dict of model name -> save success
        """
        results = {}

        if self.models_loaded.get("xgboost"):
            results["xgboost"] = self.xgboost.save()

        if self.models_loaded.get("lstm"):
            results["lstm"] = self.lstm.save()

        return results

    def predict_with_sentiment(
        self,
        df: pd.DataFrame,
        symbol: str,
        use_news: bool = True,
    ) -> dict[str, Any]:
        """Predict with automatic sentiment analysis.

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol for news fetching
            use_news: Whether to fetch and analyze news

        Returns:
            Prediction with sentiment integration
        """
        sentiment_score = None
        sentiment_data = {}

        if use_news:
            try:
                from stockai.core.sentiment import SentimentAnalyzer, NewsAggregator

                # Fetch recent news
                news_agg = NewsAggregator()
                articles = news_agg.fetch_all(symbol, max_articles=10, days_back=7)

                if articles:
                    # Analyze sentiment
                    analyzer = SentimentAnalyzer()
                    aggregated = analyzer.aggregate_sentiment(articles, symbol)

                    sentiment_score = aggregated.avg_sentiment_score
                    sentiment_data = {
                        "article_count": aggregated.article_count,
                        "dominant_label": aggregated.dominant_label.value,
                        "confidence": aggregated.confidence,
                        "signal_strength": aggregated.signal_strength,
                    }

                    logger.info(
                        f"Sentiment for {symbol}: {sentiment_score:.2f} "
                        f"({aggregated.dominant_label.value})"
                    )

            except ImportError:
                logger.warning("Sentiment module not available")
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")

        # Get prediction with sentiment
        result = self.predict(df, sentiment_score=sentiment_score)

        # Add sentiment data to result
        if sentiment_data:
            result["sentiment_data"] = sentiment_data

        return result
