"""E2E Tests for Prediction Accuracy Tracking.

Tests the prediction accuracy tracking system including:
- PredictionAccuracyTracker service
- Accuracy calculation logic
- Metrics and statistics
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from stockai.data.database import init_database, get_db
from stockai.data.models import Stock, Prediction
from stockai.core.predictor.accuracy import (
    PredictionAccuracyTracker,
    NEUTRAL_THRESHOLD_PERCENT,
    _adjust_to_business_day,
)


class TestBusinessDayAdjustment:
    """Tests for _adjust_to_business_day helper function."""

    def test_adjust_weekday_no_change(self):
        """Test that weekdays are not adjusted."""
        # Wednesday
        wednesday = datetime(2024, 1, 10, 12, 0, 0)
        assert wednesday.weekday() == 2  # Wednesday

        result = _adjust_to_business_day(wednesday, forward=True)
        assert result == wednesday

    def test_adjust_saturday_forward(self):
        """Test adjusting Saturday forward to Monday."""
        saturday = datetime(2024, 1, 6, 12, 0, 0)
        assert saturday.weekday() == 5  # Saturday

        result = _adjust_to_business_day(saturday, forward=True)
        assert result.weekday() == 0  # Monday
        assert result.date() == datetime(2024, 1, 8).date()

    def test_adjust_saturday_backward(self):
        """Test adjusting Saturday backward to Friday."""
        saturday = datetime(2024, 1, 6, 12, 0, 0)
        assert saturday.weekday() == 5  # Saturday

        result = _adjust_to_business_day(saturday, forward=False)
        assert result.weekday() == 4  # Friday
        assert result.date() == datetime(2024, 1, 5).date()

    def test_adjust_sunday_forward(self):
        """Test adjusting Sunday forward to Monday."""
        sunday = datetime(2024, 1, 7, 12, 0, 0)
        assert sunday.weekday() == 6  # Sunday

        result = _adjust_to_business_day(sunday, forward=True)
        assert result.weekday() == 0  # Monday
        assert result.date() == datetime(2024, 1, 8).date()

    def test_adjust_sunday_backward(self):
        """Test adjusting Sunday backward to Friday."""
        sunday = datetime(2024, 1, 7, 12, 0, 0)
        assert sunday.weekday() == 6  # Sunday

        result = _adjust_to_business_day(sunday, forward=False)
        assert result.weekday() == 4  # Friday
        assert result.date() == datetime(2024, 1, 5).date()


class TestPredictionAccuracyTracker:
    """Test suite for PredictionAccuracyTracker."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database and create test data."""
        init_database()
        self.tracker = PredictionAccuracyTracker()
        # Clear any existing predictions
        with get_db() as session:
            session.query(Prediction).delete()
            session.query(Stock).delete()
            session.commit()

    def _create_test_stock(self, session, symbol="BBCA", name="Bank Central Asia"):
        """Create a test stock and return it."""
        stock = Stock(symbol=symbol, name=name, sector="Finance", industry="Banking")
        session.add(stock)
        session.commit()
        session.refresh(stock)
        return stock

    def _create_test_prediction(
        self,
        session,
        stock,
        direction="UP",
        confidence=0.75,
        prediction_date=None,
        target_date=None,
        is_correct=None,
        actual_direction=None,
        actual_return=None,
        xgboost_prob=0.6,
        lstm_prob=0.5,
        sentiment_score=0.3,
    ):
        """Create a test prediction and return it."""
        if prediction_date is None:
            prediction_date = datetime.utcnow() - timedelta(days=10)
        if target_date is None:
            target_date = datetime.utcnow() - timedelta(days=5)

        prediction = Prediction(
            stock_id=stock.id,
            prediction_date=prediction_date,
            target_date=target_date,
            direction=direction,
            confidence=confidence,
            is_correct=is_correct,
            actual_direction=actual_direction,
            actual_return=actual_return,
            xgboost_prob=xgboost_prob,
            lstm_prob=lstm_prob,
            sentiment_score=sentiment_score,
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        return prediction

    def test_get_pending_predictions_empty(self):
        """Test getting pending predictions when none exist."""
        pending = self.tracker.get_pending_predictions()
        assert pending == []

    def test_get_pending_predictions_with_data(self):
        """Test getting pending predictions with past target dates."""
        with get_db() as session:
            stock = self._create_test_stock(session)
            # Create prediction with past target date and is_correct=None
            self._create_test_prediction(
                session,
                stock,
                prediction_date=datetime.utcnow() - timedelta(days=10),
                target_date=datetime.utcnow() - timedelta(days=5),
                is_correct=None,
            )

        pending = self.tracker.get_pending_predictions()
        assert len(pending) == 1
        assert pending[0]["symbol"] == "BBCA"

    def test_get_pending_predictions_excludes_evaluated(self):
        """Test that already evaluated predictions are excluded."""
        with get_db() as session:
            stock = self._create_test_stock(session)
            # Create evaluated prediction (is_correct is set)
            self._create_test_prediction(
                session,
                stock,
                target_date=datetime.utcnow() - timedelta(days=5),
                is_correct=True,
            )

        pending = self.tracker.get_pending_predictions()
        assert pending == []

    def test_get_pending_predictions_excludes_future(self):
        """Test that future predictions are excluded."""
        with get_db() as session:
            stock = self._create_test_stock(session)
            # Create prediction with future target date
            self._create_test_prediction(
                session,
                stock,
                target_date=datetime.utcnow() + timedelta(days=5),
                is_correct=None,
            )

        pending = self.tracker.get_pending_predictions()
        assert pending == []

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_calculate_actual_values_up_direction(self, mock_yahoo_class):
        """Test calculation when actual movement is UP."""
        # Create mock price data
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo

        dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
        mock_df = pd.DataFrame({
            "date": dates,
            "close": [100 + i for i in range(30)],  # Steadily increasing
        })
        mock_yahoo.get_price_history.return_value = mock_df

        tracker = PredictionAccuracyTracker()
        result = tracker._calculate_actual_values(
            symbol="BBCA",
            prediction_date=datetime(2024, 1, 5),
            target_date=datetime(2024, 1, 15),
        )

        assert result is not None
        assert result["actual_direction"] == "UP"
        assert result["actual_return"] > 0

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_calculate_actual_values_down_direction(self, mock_yahoo_class):
        """Test calculation when actual movement is DOWN."""
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo

        dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
        mock_df = pd.DataFrame({
            "date": dates,
            "close": [130 - i for i in range(30)],  # Steadily decreasing
        })
        mock_yahoo.get_price_history.return_value = mock_df

        tracker = PredictionAccuracyTracker()
        result = tracker._calculate_actual_values(
            symbol="BBCA",
            prediction_date=datetime(2024, 1, 5),
            target_date=datetime(2024, 1, 15),
        )

        assert result is not None
        assert result["actual_direction"] == "DOWN"
        assert result["actual_return"] < 0

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_calculate_actual_values_neutral_direction(self, mock_yahoo_class):
        """Test calculation when movement is within NEUTRAL threshold."""
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo

        dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
        # Prices with very small change (< 0.5%)
        mock_df = pd.DataFrame({
            "date": dates,
            "close": [100.0 + (0.001 * i) for i in range(30)],
        })
        mock_yahoo.get_price_history.return_value = mock_df

        tracker = PredictionAccuracyTracker()
        result = tracker._calculate_actual_values(
            symbol="BBCA",
            prediction_date=datetime(2024, 1, 5),
            target_date=datetime(2024, 1, 15),
        )

        assert result is not None
        assert result["actual_direction"] == "NEUTRAL"
        assert abs(result["actual_return"]) < NEUTRAL_THRESHOLD_PERCENT

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_calculate_actual_values_missing_data(self, mock_yahoo_class):
        """Test handling of missing price data."""
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo
        mock_yahoo.get_price_history.return_value = None

        tracker = PredictionAccuracyTracker()
        result = tracker._calculate_actual_values(
            symbol="BBCA",
            prediction_date=datetime(2024, 1, 5),
            target_date=datetime(2024, 1, 15),
        )

        assert result is None

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_calculate_actual_values_empty_data(self, mock_yahoo_class):
        """Test handling of empty price data."""
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo
        mock_yahoo.get_price_history.return_value = pd.DataFrame()

        tracker = PredictionAccuracyTracker()
        result = tracker._calculate_actual_values(
            symbol="BBCA",
            prediction_date=datetime(2024, 1, 5),
            target_date=datetime(2024, 1, 15),
        )

        assert result is None

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_calculate_actual_values_insufficient_data(self, mock_yahoo_class):
        """Test handling of insufficient price data."""
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo

        # Only 1 data point
        mock_df = pd.DataFrame({
            "date": [datetime(2024, 1, 10)],
            "close": [100.0],
        })
        mock_yahoo.get_price_history.return_value = mock_df

        tracker = PredictionAccuracyTracker()
        result = tracker._calculate_actual_values(
            symbol="BBCA",
            prediction_date=datetime(2024, 1, 5),
            target_date=datetime(2024, 1, 15),
        )

        assert result is None

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_update_past_predictions_no_pending(self, mock_yahoo_class):
        """Test update when no predictions are pending."""
        result = self.tracker.update_past_predictions()

        assert result["updated_count"] == 0
        assert result["message"] == "No predictions to update"

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_update_past_predictions_with_data(self, mock_yahoo_class):
        """Test updating past predictions with actual outcomes."""
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo

        # Create price data showing UP movement
        dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
        mock_df = pd.DataFrame({
            "date": dates,
            "close": [100 + (i * 2) for i in range(30)],  # UP movement
        })
        mock_yahoo.get_price_history.return_value = mock_df

        # Create pending prediction
        with get_db() as session:
            stock = self._create_test_stock(session)
            self._create_test_prediction(
                session,
                stock,
                direction="UP",
                prediction_date=datetime(2024, 1, 5),
                target_date=datetime(2024, 1, 15),
                is_correct=None,
            )

        # Create new tracker with mocked yahoo
        tracker = PredictionAccuracyTracker()
        tracker._yahoo = mock_yahoo

        result = tracker.update_past_predictions()

        assert result["updated_count"] == 1
        assert result["skipped_count"] == 0

        # Verify prediction was updated
        with get_db() as session:
            pred = session.query(Prediction).first()
            assert pred.is_correct is True
            assert pred.actual_direction == "UP"
            assert pred.actual_return > 0

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_update_past_predictions_wrong_prediction(self, mock_yahoo_class):
        """Test updating when prediction was wrong (predicted UP, went DOWN)."""
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo

        # Create price data showing DOWN movement
        dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
        mock_df = pd.DataFrame({
            "date": dates,
            "close": [130 - (i * 2) for i in range(30)],  # DOWN movement
        })
        mock_yahoo.get_price_history.return_value = mock_df

        # Create pending prediction for UP
        with get_db() as session:
            stock = self._create_test_stock(session)
            self._create_test_prediction(
                session,
                stock,
                direction="UP",
                prediction_date=datetime(2024, 1, 5),
                target_date=datetime(2024, 1, 15),
                is_correct=None,
            )

        tracker = PredictionAccuracyTracker()
        tracker._yahoo = mock_yahoo

        tracker.update_past_predictions()

        # Verify prediction was marked incorrect
        with get_db() as session:
            pred = session.query(Prediction).first()
            assert pred.is_correct is False
            assert pred.actual_direction == "DOWN"
            assert pred.actual_return < 0


class TestGetAccuracyMetrics:
    """Test suite for get_accuracy_metrics() method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database."""
        init_database()
        self.tracker = PredictionAccuracyTracker()
        with get_db() as session:
            session.query(Prediction).delete()
            session.query(Stock).delete()
            session.commit()

    def _create_test_stock(self, session, symbol="BBCA"):
        stock = Stock(symbol=symbol, name=f"{symbol} Company", sector="Finance")
        session.add(stock)
        session.commit()
        session.refresh(stock)
        return stock

    def _create_evaluated_prediction(
        self, session, stock, direction, is_correct, confidence=0.5
    ):
        """Create an evaluated prediction (is_correct is set)."""
        pred = Prediction(
            stock_id=stock.id,
            prediction_date=datetime.utcnow() - timedelta(days=10),
            target_date=datetime.utcnow() - timedelta(days=5),
            direction=direction,
            confidence=confidence,
            is_correct=is_correct,
            actual_direction=direction if is_correct else ("DOWN" if direction == "UP" else "UP"),
            actual_return=5.0 if is_correct else -5.0,
        )
        session.add(pred)
        session.commit()
        return pred

    def test_get_accuracy_metrics_empty(self):
        """Test accuracy metrics with no predictions."""
        metrics = self.tracker.get_accuracy_metrics()

        assert metrics["total_predictions"] == 0
        assert metrics["correct_predictions"] == 0
        assert metrics["accuracy_rate"] == 0.0
        assert "message" in metrics

    def test_get_accuracy_metrics_all_correct(self):
        """Test accuracy metrics when all predictions are correct."""
        with get_db() as session:
            stock = self._create_test_stock(session)
            for _ in range(5):
                self._create_evaluated_prediction(session, stock, "UP", True)

        metrics = self.tracker.get_accuracy_metrics()

        assert metrics["total_predictions"] == 5
        assert metrics["correct_predictions"] == 5
        assert metrics["accuracy_rate"] == 100.0

    def test_get_accuracy_metrics_all_wrong(self):
        """Test accuracy metrics when all predictions are wrong."""
        with get_db() as session:
            stock = self._create_test_stock(session)
            for _ in range(5):
                self._create_evaluated_prediction(session, stock, "UP", False)

        metrics = self.tracker.get_accuracy_metrics()

        assert metrics["total_predictions"] == 5
        assert metrics["correct_predictions"] == 0
        assert metrics["accuracy_rate"] == 0.0

    def test_get_accuracy_metrics_mixed(self):
        """Test accuracy metrics with mixed correct/incorrect."""
        with get_db() as session:
            stock = self._create_test_stock(session)
            # 3 correct, 2 wrong = 60% accuracy
            for _ in range(3):
                self._create_evaluated_prediction(session, stock, "UP", True)
            for _ in range(2):
                self._create_evaluated_prediction(session, stock, "UP", False)

        metrics = self.tracker.get_accuracy_metrics()

        assert metrics["total_predictions"] == 5
        assert metrics["correct_predictions"] == 3
        assert metrics["accuracy_rate"] == 60.0

    def test_get_accuracy_metrics_by_direction(self):
        """Test accuracy breakdown by direction."""
        with get_db() as session:
            stock = self._create_test_stock(session)
            # 2 correct UP, 1 wrong UP
            for _ in range(2):
                self._create_evaluated_prediction(session, stock, "UP", True)
            self._create_evaluated_prediction(session, stock, "UP", False)

            # 1 correct DOWN, 1 wrong DOWN
            self._create_evaluated_prediction(session, stock, "DOWN", True)
            self._create_evaluated_prediction(session, stock, "DOWN", False)

        metrics = self.tracker.get_accuracy_metrics()

        assert "by_direction" in metrics
        assert metrics["by_direction"]["UP"]["total"] == 3
        assert metrics["by_direction"]["UP"]["correct"] == 2
        assert abs(metrics["by_direction"]["UP"]["accuracy_rate"] - 66.67) < 0.1

        assert metrics["by_direction"]["DOWN"]["total"] == 2
        assert metrics["by_direction"]["DOWN"]["correct"] == 1
        assert metrics["by_direction"]["DOWN"]["accuracy_rate"] == 50.0

    def test_get_accuracy_metrics_by_confidence(self):
        """Test accuracy breakdown by confidence level."""
        with get_db() as session:
            stock = self._create_test_stock(session)
            # HIGH confidence (>= 0.7)
            pred = Prediction(
                stock_id=stock.id,
                prediction_date=datetime.utcnow() - timedelta(days=10),
                target_date=datetime.utcnow() - timedelta(days=5),
                direction="UP",
                confidence=0.8,
                is_correct=True,
                actual_direction="UP",
                actual_return=5.0,
            )
            session.add(pred)

            # MEDIUM confidence (0.4-0.7)
            pred2 = Prediction(
                stock_id=stock.id,
                prediction_date=datetime.utcnow() - timedelta(days=10),
                target_date=datetime.utcnow() - timedelta(days=5),
                direction="UP",
                confidence=0.5,
                is_correct=False,
                actual_direction="DOWN",
                actual_return=-5.0,
            )
            session.add(pred2)

            # LOW confidence (< 0.4)
            pred3 = Prediction(
                stock_id=stock.id,
                prediction_date=datetime.utcnow() - timedelta(days=10),
                target_date=datetime.utcnow() - timedelta(days=5),
                direction="UP",
                confidence=0.3,
                is_correct=True,
                actual_direction="UP",
                actual_return=2.0,
            )
            session.add(pred3)
            session.commit()

        metrics = self.tracker.get_accuracy_metrics()

        assert "by_confidence" in metrics
        assert metrics["by_confidence"]["HIGH"]["total"] == 1
        assert metrics["by_confidence"]["HIGH"]["correct"] == 1
        assert metrics["by_confidence"]["HIGH"]["accuracy_rate"] == 100.0

        assert metrics["by_confidence"]["MEDIUM"]["total"] == 1
        assert metrics["by_confidence"]["MEDIUM"]["correct"] == 0
        assert metrics["by_confidence"]["MEDIUM"]["accuracy_rate"] == 0.0

        assert metrics["by_confidence"]["LOW"]["total"] == 1
        assert metrics["by_confidence"]["LOW"]["correct"] == 1
        assert metrics["by_confidence"]["LOW"]["accuracy_rate"] == 100.0


class TestGetStockAccuracy:
    """Test suite for get_stock_accuracy() method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database."""
        init_database()
        self.tracker = PredictionAccuracyTracker()
        with get_db() as session:
            session.query(Prediction).delete()
            session.query(Stock).delete()
            session.commit()

    def _create_test_stock(self, session, symbol="BBCA"):
        stock = Stock(symbol=symbol, name=f"{symbol} Company", sector="Finance")
        session.add(stock)
        session.commit()
        session.refresh(stock)
        return stock

    def test_get_stock_accuracy_unknown_symbol(self):
        """Test accuracy for unknown stock symbol."""
        result = self.tracker.get_stock_accuracy("XXXX")

        assert result["symbol"] == "XXXX"
        assert result["total_predictions"] == 0
        assert "message" in result
        assert "not found" in result["message"]

    def test_get_stock_accuracy_no_predictions(self):
        """Test accuracy for stock with no predictions."""
        with get_db() as session:
            self._create_test_stock(session, "BBCA")

        result = self.tracker.get_stock_accuracy("BBCA")

        assert result["symbol"] == "BBCA"
        assert result["total_predictions"] == 0
        assert "message" in result

    def test_get_stock_accuracy_with_data(self):
        """Test accuracy for stock with predictions."""
        with get_db() as session:
            stock = self._create_test_stock(session, "BBCA")

            # Create 3 correct, 1 wrong = 75% accuracy
            for _ in range(3):
                pred = Prediction(
                    stock_id=stock.id,
                    prediction_date=datetime.utcnow() - timedelta(days=10),
                    target_date=datetime.utcnow() - timedelta(days=5),
                    direction="UP",
                    confidence=0.7,
                    is_correct=True,
                    actual_direction="UP",
                    actual_return=5.0,
                )
                session.add(pred)

            pred = Prediction(
                stock_id=stock.id,
                prediction_date=datetime.utcnow() - timedelta(days=10),
                target_date=datetime.utcnow() - timedelta(days=5),
                direction="UP",
                confidence=0.6,
                is_correct=False,
                actual_direction="DOWN",
                actual_return=-3.0,
            )
            session.add(pred)
            session.commit()

        result = self.tracker.get_stock_accuracy("BBCA")

        assert result["symbol"] == "BBCA"
        assert result["total_predictions"] == 4
        assert result["correct_predictions"] == 3
        assert result["accuracy_rate"] == 75.0

    def test_get_stock_accuracy_includes_recent_predictions(self):
        """Test that recent predictions are included."""
        with get_db() as session:
            stock = self._create_test_stock(session, "BBCA")

            pred = Prediction(
                stock_id=stock.id,
                prediction_date=datetime.utcnow() - timedelta(days=10),
                target_date=datetime.utcnow() - timedelta(days=5),
                direction="UP",
                confidence=0.8,
                is_correct=True,
                actual_direction="UP",
                actual_return=5.5,
            )
            session.add(pred)
            session.commit()

        result = self.tracker.get_stock_accuracy("BBCA")

        assert "recent_predictions" in result
        assert len(result["recent_predictions"]) == 1
        assert result["recent_predictions"][0]["direction"] == "UP"
        assert result["recent_predictions"][0]["is_correct"] is True

    def test_get_stock_accuracy_includes_trend(self):
        """Test that accuracy trend is included."""
        with get_db() as session:
            stock = self._create_test_stock(session, "BBCA")

            # Create predictions for different months
            for month in range(1, 4):
                pred = Prediction(
                    stock_id=stock.id,
                    prediction_date=datetime(2024, month, 1),
                    target_date=datetime(2024, month, 15),
                    direction="UP",
                    confidence=0.7,
                    is_correct=True,
                    actual_direction="UP",
                    actual_return=5.0,
                )
                session.add(pred)
            session.commit()

        result = self.tracker.get_stock_accuracy("BBCA")

        assert "accuracy_trend" in result
        assert len(result["accuracy_trend"]) == 3


class TestGetAccuracyByModel:
    """Test suite for get_accuracy_by_model() method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database."""
        init_database()
        self.tracker = PredictionAccuracyTracker()
        with get_db() as session:
            session.query(Prediction).delete()
            session.query(Stock).delete()
            session.commit()

    def _create_test_stock(self, session, symbol="BBCA"):
        stock = Stock(symbol=symbol, name=f"{symbol} Company", sector="Finance")
        session.add(stock)
        session.commit()
        session.refresh(stock)
        return stock

    def test_get_accuracy_by_model_empty(self):
        """Test model accuracy analysis with no predictions."""
        result = self.tracker.get_accuracy_by_model()

        assert result["xgboost"]["bins"] == []
        assert result["lstm"]["bins"] == []
        assert result["sentiment"]["bins"] == []
        assert "message" in result

    def test_get_accuracy_by_model_with_data(self):
        """Test model accuracy analysis with predictions."""
        with get_db() as session:
            stock = self._create_test_stock(session, "BBCA")

            # Create predictions with varying model probabilities
            for i in range(10):
                pred = Prediction(
                    stock_id=stock.id,
                    prediction_date=datetime.utcnow() - timedelta(days=10),
                    target_date=datetime.utcnow() - timedelta(days=5),
                    direction="UP",
                    confidence=0.7,
                    is_correct=(i % 2 == 0),  # 50% correct
                    actual_direction="UP" if (i % 2 == 0) else "DOWN",
                    actual_return=5.0 if (i % 2 == 0) else -5.0,
                    xgboost_prob=0.1 * i,  # 0.0 to 0.9
                    lstm_prob=0.5,
                    sentiment_score=0.0,
                )
                session.add(pred)
            session.commit()

        result = self.tracker.get_accuracy_by_model()

        assert result["xgboost"]["total_with_data"] == 10
        assert len(result["xgboost"]["bins"]) == 5
        assert "correlation_summary" in result
        assert "insights" in result
        assert isinstance(result["insights"], list)

    def test_get_accuracy_by_model_correlation_summary(self):
        """Test that correlation summary is calculated."""
        with get_db() as session:
            stock = self._create_test_stock(session, "BBCA")

            # Create predictions where high xgboost = correct
            for i in range(10):
                is_high_prob = i >= 5
                pred = Prediction(
                    stock_id=stock.id,
                    prediction_date=datetime.utcnow() - timedelta(days=10),
                    target_date=datetime.utcnow() - timedelta(days=5),
                    direction="UP",
                    confidence=0.7,
                    is_correct=is_high_prob,  # High prob = correct
                    actual_direction="UP" if is_high_prob else "DOWN",
                    actual_return=5.0 if is_high_prob else -5.0,
                    xgboost_prob=0.1 * (i + 1),  # 0.1 to 1.0
                    lstm_prob=0.5,
                    sentiment_score=0.0,
                )
                session.add(pred)
            session.commit()

        result = self.tracker.get_accuracy_by_model()

        assert "correlation_summary" in result
        assert "xgboost" in result["correlation_summary"]
        # High probs are always correct, low are always wrong - should show correlation


class TestPredictionAccuracyTrackerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database."""
        init_database()
        self.tracker = PredictionAccuracyTracker()

    def test_tracker_with_injected_session(self):
        """Test tracker with injected session."""
        with get_db() as session:
            tracker = PredictionAccuracyTracker(session=session)
            assert tracker._session is session
            assert tracker._use_context_manager is False

    def test_tracker_without_session(self):
        """Test tracker creates its own session."""
        tracker = PredictionAccuracyTracker()
        assert tracker._session is None
        assert tracker._use_context_manager is True

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_calculate_actual_values_exception_handling(self, mock_yahoo_class):
        """Test exception handling in calculate_actual_values."""
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo
        mock_yahoo.get_price_history.side_effect = Exception("Network error")

        tracker = PredictionAccuracyTracker()
        result = tracker._calculate_actual_values(
            symbol="BBCA",
            prediction_date=datetime(2024, 1, 5),
            target_date=datetime(2024, 1, 15),
        )

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
