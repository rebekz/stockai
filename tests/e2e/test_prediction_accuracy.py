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

from stockai.data.database import init_database, session_scope
from stockai.data.models import Stock, Prediction, StockPrice, PortfolioItem, PortfolioTransaction, WatchlistItem, NewsArticle
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
        with session_scope() as session:
            session.query(NewsArticle).delete()
            session.query(WatchlistItem).delete()
            session.query(PortfolioTransaction).delete()
            session.query(PortfolioItem).delete()
            session.query(StockPrice).delete()
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
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
            session.query(NewsArticle).delete()
            session.query(WatchlistItem).delete()
            session.query(PortfolioTransaction).delete()
            session.query(PortfolioItem).delete()
            session.query(StockPrice).delete()
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
        with session_scope() as session:
            stock = self._create_test_stock(session)
            for _ in range(5):
                self._create_evaluated_prediction(session, stock, "UP", True)

        metrics = self.tracker.get_accuracy_metrics()

        assert metrics["total_predictions"] == 5
        assert metrics["correct_predictions"] == 5
        assert metrics["accuracy_rate"] == 100.0

    def test_get_accuracy_metrics_all_wrong(self):
        """Test accuracy metrics when all predictions are wrong."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            for _ in range(5):
                self._create_evaluated_prediction(session, stock, "UP", False)

        metrics = self.tracker.get_accuracy_metrics()

        assert metrics["total_predictions"] == 5
        assert metrics["correct_predictions"] == 0
        assert metrics["accuracy_rate"] == 0.0

    def test_get_accuracy_metrics_mixed(self):
        """Test accuracy metrics with mixed correct/incorrect."""
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
            session.query(NewsArticle).delete()
            session.query(WatchlistItem).delete()
            session.query(PortfolioTransaction).delete()
            session.query(PortfolioItem).delete()
            session.query(StockPrice).delete()
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
        with session_scope() as session:
            self._create_test_stock(session, "BBCA")

        result = self.tracker.get_stock_accuracy("BBCA")

        assert result["symbol"] == "BBCA"
        assert result["total_predictions"] == 0
        assert "message" in result

    def test_get_stock_accuracy_with_data(self):
        """Test accuracy for stock with predictions."""
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
            session.query(NewsArticle).delete()
            session.query(WatchlistItem).delete()
            session.query(PortfolioTransaction).delete()
            session.query(PortfolioItem).delete()
            session.query(StockPrice).delete()
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
        with session_scope() as session:
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
        with session_scope() as session:
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
        with session_scope() as session:
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


# ============ API Endpoint Tests ============

from fastapi.testclient import TestClient
from stockai.web.app import create_app


class TestPredictionAccuracyAPI:
    """Tests for prediction accuracy API endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database and client."""
        init_database()
        app = create_app()
        self.client = TestClient(app)
        # Clear existing data
        with session_scope() as session:
            session.query(NewsArticle).delete()
            session.query(WatchlistItem).delete()
            session.query(PortfolioTransaction).delete()
            session.query(PortfolioItem).delete()
            session.query(StockPrice).delete()
            session.query(Prediction).delete()
            session.query(Stock).delete()
            session.commit()

    def _create_test_stock(self, session, symbol="BBCA"):
        """Create a test stock."""
        stock = Stock(symbol=symbol, name=f"{symbol} Company", sector="Finance")
        session.add(stock)
        session.commit()
        session.refresh(stock)
        return stock

    def _create_evaluated_prediction(
        self, session, stock, direction="UP", is_correct=True, confidence=0.7
    ):
        """Create an evaluated prediction."""
        pred = Prediction(
            stock_id=stock.id,
            prediction_date=datetime.utcnow() - timedelta(days=10),
            target_date=datetime.utcnow() - timedelta(days=5),
            direction=direction,
            confidence=confidence,
            is_correct=is_correct,
            actual_direction=direction if is_correct else ("DOWN" if direction == "UP" else "UP"),
            actual_return=5.0 if is_correct else -5.0,
            xgboost_prob=0.6,
            lstm_prob=0.5,
            sentiment_score=0.3,
        )
        session.add(pred)
        session.commit()
        return pred

    def _create_pending_prediction(self, session, stock, direction="UP", confidence=0.7):
        """Create a pending prediction (not yet evaluated)."""
        pred = Prediction(
            stock_id=stock.id,
            prediction_date=datetime.utcnow() - timedelta(days=10),
            target_date=datetime.utcnow() - timedelta(days=5),
            direction=direction,
            confidence=confidence,
            is_correct=None,
            actual_direction=None,
            actual_return=None,
            xgboost_prob=0.6,
            lstm_prob=0.5,
            sentiment_score=0.3,
        )
        session.add(pred)
        session.commit()
        return pred


class TestGetPredictionAccuracyEndpoint(TestPredictionAccuracyAPI):
    """Tests for GET /api/predictions/accuracy endpoint."""

    def test_get_accuracy_returns_200(self):
        """Endpoint should return 200 OK."""
        response = self.client.get("/api/predictions/accuracy")
        assert response.status_code == 200

    def test_get_accuracy_empty_database(self):
        """Should return valid metrics when no predictions exist."""
        response = self.client.get("/api/predictions/accuracy")
        assert response.status_code == 200

        data = response.json()
        assert data["total_predictions"] == 0
        assert data["correct_predictions"] == 0
        assert data["accuracy_rate"] == 0.0
        assert "message" in data

    def test_get_accuracy_with_predictions(self):
        """Should return accuracy metrics when predictions exist."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            # 3 correct, 2 wrong = 60% accuracy
            for _ in range(3):
                self._create_evaluated_prediction(session, stock, is_correct=True)
            for _ in range(2):
                self._create_evaluated_prediction(session, stock, is_correct=False)

        response = self.client.get("/api/predictions/accuracy")
        assert response.status_code == 200

        data = response.json()
        assert data["total_predictions"] == 5
        assert data["correct_predictions"] == 3
        assert data["accuracy_rate"] == 60.0

    def test_get_accuracy_includes_direction_breakdown(self):
        """Should include accuracy breakdown by direction."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            self._create_evaluated_prediction(session, stock, direction="UP", is_correct=True)
            self._create_evaluated_prediction(session, stock, direction="DOWN", is_correct=False)

        response = self.client.get("/api/predictions/accuracy")
        data = response.json()

        assert "by_direction" in data
        assert "UP" in data["by_direction"]
        assert "DOWN" in data["by_direction"]

    def test_get_accuracy_includes_confidence_breakdown(self):
        """Should include accuracy breakdown by confidence level."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            # HIGH confidence
            self._create_evaluated_prediction(session, stock, confidence=0.8, is_correct=True)
            # MEDIUM confidence
            self._create_evaluated_prediction(session, stock, confidence=0.5, is_correct=False)
            # LOW confidence
            self._create_evaluated_prediction(session, stock, confidence=0.3, is_correct=True)

        response = self.client.get("/api/predictions/accuracy")
        data = response.json()

        assert "by_confidence" in data
        assert "HIGH" in data["by_confidence"]
        assert "MEDIUM" in data["by_confidence"]
        assert "LOW" in data["by_confidence"]

    def test_get_accuracy_json_content_type(self):
        """Endpoint should return JSON."""
        response = self.client.get("/api/predictions/accuracy")
        assert "application/json" in response.headers["content-type"]


class TestGetStockAccuracyEndpoint(TestPredictionAccuracyAPI):
    """Tests for GET /api/predictions/accuracy/{symbol} endpoint."""

    def test_get_stock_accuracy_unknown_symbol(self):
        """Should return 404 for unknown stock symbol."""
        response = self.client.get("/api/predictions/accuracy/UNKNOWN")
        assert response.status_code == 404

    def test_get_stock_accuracy_no_predictions(self):
        """Should return 404 for stock with no predictions."""
        with session_scope() as session:
            self._create_test_stock(session, "BBCA")

        response = self.client.get("/api/predictions/accuracy/BBCA")
        assert response.status_code == 404

    def test_get_stock_accuracy_with_data(self):
        """Should return accuracy metrics for stock with predictions."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            # 3 correct, 1 wrong = 75% accuracy
            for _ in range(3):
                self._create_evaluated_prediction(session, stock, is_correct=True)
            self._create_evaluated_prediction(session, stock, is_correct=False)

        response = self.client.get("/api/predictions/accuracy/BBCA")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "BBCA"
        assert data["total_predictions"] == 4
        assert data["correct_predictions"] == 3
        assert data["accuracy_rate"] == 75.0

    def test_get_stock_accuracy_case_insensitive(self):
        """Should handle lowercase symbol."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            self._create_evaluated_prediction(session, stock, is_correct=True)

        response = self.client.get("/api/predictions/accuracy/bbca")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "BBCA"

    def test_get_stock_accuracy_includes_recent_predictions(self):
        """Should include recent predictions in response."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            self._create_evaluated_prediction(session, stock, direction="UP", is_correct=True)

        response = self.client.get("/api/predictions/accuracy/BBCA")
        data = response.json()

        assert "recent_predictions" in data
        assert len(data["recent_predictions"]) >= 1
        assert data["recent_predictions"][0]["direction"] == "UP"
        assert data["recent_predictions"][0]["is_correct"] is True

    def test_get_stock_accuracy_includes_trend(self):
        """Should include accuracy trend in response."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            # Create predictions for different months
            for month in [1, 2, 3]:
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

        response = self.client.get("/api/predictions/accuracy/BBCA")
        data = response.json()

        assert "accuracy_trend" in data
        assert len(data["accuracy_trend"]) == 3

    def test_get_stock_accuracy_includes_breakdown(self):
        """Should include breakdowns by direction and confidence."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            self._create_evaluated_prediction(session, stock, direction="UP", is_correct=True)
            self._create_evaluated_prediction(session, stock, direction="DOWN", is_correct=False)

        response = self.client.get("/api/predictions/accuracy/BBCA")
        data = response.json()

        assert "by_direction" in data
        assert "by_confidence" in data


class TestBackfillPredictionAccuracyEndpoint(TestPredictionAccuracyAPI):
    """Tests for POST /api/predictions/backfill endpoint."""

    def test_backfill_returns_200(self):
        """Endpoint should return 200 OK."""
        response = self.client.post("/api/predictions/backfill")
        assert response.status_code == 200

    def test_backfill_no_pending_predictions(self):
        """Should return success when no predictions to update."""
        response = self.client.post("/api/predictions/backfill")
        assert response.status_code == 200

        data = response.json()
        assert data["updated_count"] == 0
        assert data["message"] == "No predictions to update"

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_backfill_updates_predictions(self, mock_yahoo_class):
        """Should update pending predictions with actual data."""
        # Setup mock
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo

        dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
        mock_df = pd.DataFrame({
            "date": dates,
            "close": [100 + (i * 2) for i in range(30)],  # UP movement
        })
        mock_yahoo.get_price_history.return_value = mock_df

        # Create pending prediction
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            pred = Prediction(
                stock_id=stock.id,
                prediction_date=datetime(2024, 1, 5),
                target_date=datetime(2024, 1, 15),
                direction="UP",
                confidence=0.7,
                is_correct=None,
                actual_direction=None,
                actual_return=None,
            )
            session.add(pred)
            session.commit()

        response = self.client.post("/api/predictions/backfill")
        assert response.status_code == 200

        data = response.json()
        assert data["updated_count"] == 1

    def test_backfill_returns_statistics(self):
        """Should return backfill statistics in response."""
        response = self.client.post("/api/predictions/backfill")
        data = response.json()

        # Should include standard statistics fields
        assert "updated_count" in data
        assert "message" in data

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_backfill_handles_missing_price_data(self, mock_yahoo_class):
        """Should handle missing price data gracefully."""
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo
        mock_yahoo.get_price_history.return_value = None

        # Create pending prediction
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            self._create_pending_prediction(session, stock)

        response = self.client.post("/api/predictions/backfill")
        assert response.status_code == 200

        data = response.json()
        assert data["skipped_count"] >= 1

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_backfill_multiple_predictions(self, mock_yahoo_class):
        """Should handle multiple pending predictions."""
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo

        dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
        mock_df = pd.DataFrame({
            "date": dates,
            "close": [100 + (i * 2) for i in range(30)],
        })
        mock_yahoo.get_price_history.return_value = mock_df

        # Create multiple pending predictions
        with session_scope() as session:
            stock1 = self._create_test_stock(session, "BBCA")
            stock2 = self._create_test_stock(session, "BBRI")

            for stock in [stock1, stock2]:
                pred = Prediction(
                    stock_id=stock.id,
                    prediction_date=datetime(2024, 1, 5),
                    target_date=datetime(2024, 1, 15),
                    direction="UP",
                    confidence=0.7,
                    is_correct=None,
                    actual_direction=None,
                    actual_return=None,
                )
                session.add(pred)
            session.commit()

        response = self.client.post("/api/predictions/backfill")
        assert response.status_code == 200

        data = response.json()
        assert data["updated_count"] == 2

    def test_backfill_json_content_type(self):
        """Endpoint should return JSON."""
        response = self.client.post("/api/predictions/backfill")
        assert "application/json" in response.headers["content-type"]


class TestPredictionAccuracyAPIEdgeCases(TestPredictionAccuracyAPI):
    """Edge case tests for prediction accuracy API endpoints."""

    def test_accuracy_with_all_correct_predictions(self):
        """Should handle 100% accuracy."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            for _ in range(5):
                self._create_evaluated_prediction(session, stock, is_correct=True)

        response = self.client.get("/api/predictions/accuracy")
        data = response.json()

        assert data["accuracy_rate"] == 100.0

    def test_accuracy_with_all_wrong_predictions(self):
        """Should handle 0% accuracy."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            for _ in range(5):
                self._create_evaluated_prediction(session, stock, is_correct=False)

        response = self.client.get("/api/predictions/accuracy")
        data = response.json()

        assert data["accuracy_rate"] == 0.0

    def test_stock_accuracy_symbol_with_suffix(self):
        """Should handle symbol with market suffix (e.g., BBCA.JK)."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA.JK")
            self._create_evaluated_prediction(session, stock, is_correct=True)

        response = self.client.get("/api/predictions/accuracy/BBCA.JK")
        assert response.status_code == 200

        data = response.json()
        assert data["symbol"] == "BBCA.JK"

    def test_backfill_idempotent(self):
        """Backfill should be idempotent - already evaluated predictions not re-processed."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            # Create already evaluated prediction
            self._create_evaluated_prediction(session, stock, is_correct=True)

        # First backfill
        response1 = self.client.post("/api/predictions/backfill")
        data1 = response1.json()

        # Second backfill
        response2 = self.client.post("/api/predictions/backfill")
        data2 = response2.json()

        # Both should return no updates
        assert data1["updated_count"] == 0
        assert data2["updated_count"] == 0


# ============ CLI Command Tests ============

from typer.testing import CliRunner
from stockai.cli.main import app

runner = CliRunner()


class TestPredictionsCLI:
    """Base class for predictions CLI tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database."""
        init_database()
        with session_scope() as session:
            session.query(NewsArticle).delete()
            session.query(WatchlistItem).delete()
            session.query(PortfolioTransaction).delete()
            session.query(PortfolioItem).delete()
            session.query(StockPrice).delete()
            session.query(Prediction).delete()
            session.query(Stock).delete()
            session.commit()

    def _create_test_stock(self, session, symbol="BBCA"):
        """Create a test stock."""
        stock = Stock(symbol=symbol, name=f"{symbol} Company", sector="Finance")
        session.add(stock)
        session.commit()
        session.refresh(stock)
        return stock

    def _create_evaluated_prediction(
        self, session, stock, direction="UP", is_correct=True, confidence=0.7
    ):
        """Create an evaluated prediction."""
        pred = Prediction(
            stock_id=stock.id,
            prediction_date=datetime.utcnow() - timedelta(days=10),
            target_date=datetime.utcnow() - timedelta(days=5),
            direction=direction,
            confidence=confidence,
            is_correct=is_correct,
            actual_direction=direction if is_correct else ("DOWN" if direction == "UP" else "UP"),
            actual_return=5.0 if is_correct else -5.0,
            xgboost_prob=0.6,
            lstm_prob=0.5,
            sentiment_score=0.3,
        )
        session.add(pred)
        session.commit()
        return pred

    def _create_pending_prediction(self, session, stock, direction="UP", confidence=0.7):
        """Create a pending prediction (not yet evaluated)."""
        pred = Prediction(
            stock_id=stock.id,
            prediction_date=datetime.utcnow() - timedelta(days=10),
            target_date=datetime.utcnow() - timedelta(days=5),
            direction=direction,
            confidence=confidence,
            is_correct=None,
            actual_direction=None,
            actual_return=None,
            xgboost_prob=0.6,
            lstm_prob=0.5,
            sentiment_score=0.3,
        )
        session.add(pred)
        session.commit()
        return pred


class TestPredictionsAccuracyCommand(TestPredictionsCLI):
    """Tests for 'stock predictions accuracy' CLI command."""

    def test_accuracy_command_exists(self):
        """Test that the accuracy command exists."""
        result = runner.invoke(app, ["predictions", "accuracy", "--help"])
        assert result.exit_code == 0
        assert "accuracy" in result.stdout.lower()

    def test_accuracy_command_no_predictions(self):
        """Test accuracy command with empty database."""
        result = runner.invoke(app, ["predictions", "accuracy"])
        assert result.exit_code == 0
        assert "No evaluated predictions found" in result.stdout or "No predictions" in result.stdout or "0" in result.stdout

    def test_accuracy_command_with_predictions(self):
        """Test accuracy command with predictions in database."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            # Create 3 correct, 2 wrong predictions
            for _ in range(3):
                self._create_evaluated_prediction(session, stock, is_correct=True)
            for _ in range(2):
                self._create_evaluated_prediction(session, stock, is_correct=False)

        result = runner.invoke(app, ["predictions", "accuracy"])
        assert result.exit_code == 0
        # Should show total predictions
        assert "5" in result.stdout or "Total" in result.stdout

    def test_accuracy_command_shows_direction_breakdown(self):
        """Test that accuracy command shows direction breakdown."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            self._create_evaluated_prediction(session, stock, direction="UP", is_correct=True)
            self._create_evaluated_prediction(session, stock, direction="DOWN", is_correct=False)

        result = runner.invoke(app, ["predictions", "accuracy"])
        assert result.exit_code == 0
        # Should show direction breakdown
        assert "UP" in result.stdout or "Direction" in result.stdout

    def test_accuracy_command_shows_confidence_breakdown(self):
        """Test that accuracy command shows confidence level breakdown."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            self._create_evaluated_prediction(session, stock, confidence=0.8, is_correct=True)
            self._create_evaluated_prediction(session, stock, confidence=0.5, is_correct=False)

        result = runner.invoke(app, ["predictions", "accuracy"])
        assert result.exit_code == 0
        # Should show confidence breakdown
        assert "HIGH" in result.stdout or "MEDIUM" in result.stdout or "Confidence" in result.stdout


class TestPredictionsAccuracySymbolOption(TestPredictionsCLI):
    """Tests for 'stock predictions accuracy --symbol' option."""

    def test_accuracy_symbol_option_unknown_symbol(self):
        """Test --symbol option with unknown stock."""
        result = runner.invoke(app, ["predictions", "accuracy", "--symbol", "UNKNOWN"])
        assert result.exit_code == 0
        # Should show message about no predictions
        assert "UNKNOWN" in result.stdout.upper()

    def test_accuracy_symbol_option_no_predictions(self):
        """Test --symbol option for stock with no predictions."""
        with session_scope() as session:
            self._create_test_stock(session, "BBCA")

        result = runner.invoke(app, ["predictions", "accuracy", "--symbol", "BBCA"])
        assert result.exit_code == 0
        assert "BBCA" in result.stdout.upper()

    def test_accuracy_symbol_option_with_predictions(self):
        """Test --symbol option for stock with predictions."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            for _ in range(3):
                self._create_evaluated_prediction(session, stock, is_correct=True)
            self._create_evaluated_prediction(session, stock, is_correct=False)

        result = runner.invoke(app, ["predictions", "accuracy", "--symbol", "BBCA"])
        assert result.exit_code == 0
        assert "BBCA" in result.stdout.upper()
        # Should show some metrics
        assert "4" in result.stdout or "Total" in result.stdout

    def test_accuracy_symbol_option_short_form(self):
        """Test -s short form of --symbol option."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBRI")
            self._create_evaluated_prediction(session, stock, is_correct=True)

        result = runner.invoke(app, ["predictions", "accuracy", "-s", "BBRI"])
        assert result.exit_code == 0
        assert "BBRI" in result.stdout.upper()

    def test_accuracy_symbol_option_case_insensitive(self):
        """Test --symbol option is case insensitive."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BMRI")
            self._create_evaluated_prediction(session, stock, is_correct=True)

        result = runner.invoke(app, ["predictions", "accuracy", "--symbol", "bmri"])
        assert result.exit_code == 0
        assert "BMRI" in result.stdout.upper()

    def test_accuracy_symbol_option_shows_recent_predictions(self):
        """Test that --symbol shows recent predictions."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            self._create_evaluated_prediction(session, stock, direction="UP", is_correct=True)

        result = runner.invoke(app, ["predictions", "accuracy", "--symbol", "BBCA"])
        assert result.exit_code == 0
        # Should show recent predictions table
        assert "Recent" in result.stdout or "Predicted" in result.stdout or "Actual" in result.stdout


class TestPredictionsAccuracyVerboseOption(TestPredictionsCLI):
    """Tests for 'stock predictions accuracy --verbose' option."""

    def test_accuracy_verbose_option(self):
        """Test --verbose option shows additional details."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            for _ in range(5):
                self._create_evaluated_prediction(session, stock, is_correct=True)

        result = runner.invoke(app, ["predictions", "accuracy", "--verbose"])
        assert result.exit_code == 0
        # Verbose should show model analysis
        assert "Model" in result.stdout or "XGBoost" in result.stdout or "Correlation" in result.stdout

    def test_accuracy_verbose_short_form(self):
        """Test -v short form of --verbose option."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            for _ in range(3):
                self._create_evaluated_prediction(session, stock, is_correct=True)

        result = runner.invoke(app, ["predictions", "accuracy", "-v"])
        assert result.exit_code == 0

    def test_accuracy_verbose_with_symbol(self):
        """Test --verbose and --symbol options together."""
        with session_scope() as session:
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

        result = runner.invoke(app, ["predictions", "accuracy", "--symbol", "BBCA", "--verbose"])
        assert result.exit_code == 0
        assert "BBCA" in result.stdout.upper()
        # Should show trend information
        assert "Trend" in result.stdout or "Monthly" in result.stdout or "2024" in result.stdout


class TestPredictionsBackfillCommand(TestPredictionsCLI):
    """Tests for 'stock predictions backfill' CLI command."""

    def test_backfill_command_exists(self):
        """Test that the backfill command exists."""
        result = runner.invoke(app, ["predictions", "backfill", "--help"])
        assert result.exit_code == 0
        assert "backfill" in result.stdout.lower()

    def test_backfill_command_no_pending_predictions(self):
        """Test backfill command with no pending predictions."""
        result = runner.invoke(app, ["predictions", "backfill"])
        assert result.exit_code == 0
        # Should indicate no predictions to update
        assert "No pending" in result.stdout or "0" in result.stdout

    @patch("stockai.core.predictor.accuracy.YahooFinanceSource")
    def test_backfill_command_with_pending_predictions(self, mock_yahoo_class):
        """Test backfill command with pending predictions."""
        # Setup mock
        mock_yahoo = MagicMock()
        mock_yahoo_class.return_value = mock_yahoo

        dates = pd.date_range(start="2024-01-01", periods=30, freq="B")
        mock_df = pd.DataFrame({
            "date": dates,
            "close": [100 + (i * 2) for i in range(30)],
        })
        mock_yahoo.get_price_history.return_value = mock_df

        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            pred = Prediction(
                stock_id=stock.id,
                prediction_date=datetime(2024, 1, 5),
                target_date=datetime(2024, 1, 15),
                direction="UP",
                confidence=0.7,
                is_correct=None,
            )
            session.add(pred)
            session.commit()

        result = runner.invoke(app, ["predictions", "backfill"])
        assert result.exit_code == 0
        # Should show some update message
        assert "1" in result.stdout or "Update" in result.stdout or "Result" in result.stdout

    def test_backfill_command_shows_results_panel(self):
        """Test backfill command shows results panel."""
        result = runner.invoke(app, ["predictions", "backfill"])
        assert result.exit_code == 0
        # Should show a results panel/message
        assert "Backfill" in result.stdout or "Complete" in result.stdout or "Result" in result.stdout


class TestPredictionsBackfillDryRunOption(TestPredictionsCLI):
    """Tests for 'stock predictions backfill --dry-run' option."""

    def test_backfill_dry_run_option_no_pending(self):
        """Test --dry-run option with no pending predictions."""
        result = runner.invoke(app, ["predictions", "backfill", "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run" in result.stdout or "no" in result.stdout.lower()

    def test_backfill_dry_run_with_pending_predictions(self):
        """Test --dry-run option shows what would be updated."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            self._create_pending_prediction(session, stock, direction="UP")

        result = runner.invoke(app, ["predictions", "backfill", "--dry-run"])
        assert result.exit_code == 0
        # Should mention dry run mode
        assert "Dry run" in result.stdout or "no changes" in result.stdout.lower()
        # Should show BBCA in the pending list
        assert "BBCA" in result.stdout.upper()

    def test_backfill_dry_run_short_form(self):
        """Test -n short form of --dry-run option."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "TLKM")
            self._create_pending_prediction(session, stock)

        result = runner.invoke(app, ["predictions", "backfill", "-n"])
        assert result.exit_code == 0
        assert "Dry run" in result.stdout or "no changes" in result.stdout.lower()

    def test_backfill_dry_run_does_not_modify_database(self):
        """Test that --dry-run does not modify the database."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            pred = self._create_pending_prediction(session, stock)
            pred_id = pred.id

        # Run dry-run
        result = runner.invoke(app, ["predictions", "backfill", "--dry-run"])
        assert result.exit_code == 0

        # Verify prediction is still pending (not updated)
        with session_scope() as session:
            pred = session.query(Prediction).filter(Prediction.id == pred_id).first()
            assert pred.is_correct is None
            assert pred.actual_direction is None
            assert pred.actual_return is None

    def test_backfill_dry_run_shows_pending_count(self):
        """Test that --dry-run shows count of pending predictions."""
        with session_scope() as session:
            stock = self._create_test_stock(session, "BBCA")
            for _ in range(3):
                self._create_pending_prediction(session, stock)

        result = runner.invoke(app, ["predictions", "backfill", "--dry-run"])
        assert result.exit_code == 0
        # Should show the count
        assert "3" in result.stdout

    def test_backfill_dry_run_shows_symbol_summary(self):
        """Test that --dry-run shows summary by symbol."""
        with session_scope() as session:
            stock1 = self._create_test_stock(session, "BBCA")
            stock2 = self._create_test_stock(session, "BBRI")
            for _ in range(2):
                self._create_pending_prediction(session, stock1)
            self._create_pending_prediction(session, stock2)

        result = runner.invoke(app, ["predictions", "backfill", "--dry-run"])
        assert result.exit_code == 0
        # Should show both symbols
        assert "BBCA" in result.stdout.upper()
        assert "BBRI" in result.stdout.upper()


class TestPredictionsCLIEdgeCases(TestPredictionsCLI):
    """Edge case tests for predictions CLI commands."""

    def test_accuracy_with_100_percent_accuracy(self):
        """Test accuracy command with 100% accuracy."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            for _ in range(5):
                self._create_evaluated_prediction(session, stock, is_correct=True)

        result = runner.invoke(app, ["predictions", "accuracy"])
        assert result.exit_code == 0
        assert "100" in result.stdout

    def test_accuracy_with_0_percent_accuracy(self):
        """Test accuracy command with 0% accuracy."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            for _ in range(5):
                self._create_evaluated_prediction(session, stock, is_correct=False)

        result = runner.invoke(app, ["predictions", "accuracy"])
        assert result.exit_code == 0
        assert "0" in result.stdout

    def test_accuracy_multiple_stocks(self):
        """Test accuracy command with multiple stocks."""
        with session_scope() as session:
            stock1 = self._create_test_stock(session, "BBCA")
            stock2 = self._create_test_stock(session, "BBRI")
            self._create_evaluated_prediction(session, stock1, is_correct=True)
            self._create_evaluated_prediction(session, stock2, is_correct=False)

        result = runner.invoke(app, ["predictions", "accuracy"])
        assert result.exit_code == 0
        # Should show overall metrics
        assert "2" in result.stdout or "Total" in result.stdout

    def test_backfill_already_evaluated_predictions(self):
        """Test backfill skips already evaluated predictions."""
        with session_scope() as session:
            stock = self._create_test_stock(session)
            # Create only evaluated predictions (not pending)
            for _ in range(3):
                self._create_evaluated_prediction(session, stock, is_correct=True)

        result = runner.invoke(app, ["predictions", "backfill"])
        assert result.exit_code == 0
        # Should show no predictions to update
        assert "No pending" in result.stdout or "0" in result.stdout

    def test_predictions_subcommand_help(self):
        """Test predictions subcommand shows help."""
        result = runner.invoke(app, ["predictions", "--help"])
        assert result.exit_code == 0
        assert "accuracy" in result.stdout
        assert "backfill" in result.stdout


class TestPredictShowAccuracyOption(TestPredictionsCLI):
    """Tests for 'stock predict --show-accuracy' option."""

    def test_predict_show_accuracy_option_exists(self):
        """Test that --show-accuracy option exists."""
        result = runner.invoke(app, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--show-accuracy" in result.stdout or "-a" in result.stdout

    def test_predict_show_accuracy_no_data(self):
        """Test --show-accuracy with no accuracy data."""
        # Create a stock but no predictions
        with session_scope() as session:
            self._create_test_stock(session, symbol="TSTX")

        # Mocking the prediction to avoid needing actual models
        with patch("stockai.cli.main.EnsemblePredictor") as mock_predictor, \
             patch("stockai.cli.main.YahooFinanceSource") as mock_yahoo:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock predictor
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": True, "lstm": True}
            mock_instance.predict.return_value = {
                "direction": "UP",
                "probability": 0.7,
                "confidence": 0.75,
                "confidence_level": "HIGH",
                "model_agreement": True,
                "contributions": {},
            }
            mock_predictor.return_value = mock_instance

            result = runner.invoke(app, ["predict", "TSTX", "--show-accuracy"])

        # Should show prediction and accuracy panel (even if no data)
        assert result.exit_code == 0
        assert "TSTX" in result.stdout.upper()
        # Should show accuracy section even with no data
        assert "Accuracy" in result.stdout or "accuracy" in result.stdout.lower()

    def test_predict_show_accuracy_with_data(self):
        """Test --show-accuracy with accuracy data available."""
        with session_scope() as session:
            stock = self._create_test_stock(session, symbol="TSTA")
            # Create some evaluated predictions for this stock
            for i in range(5):
                self._create_evaluated_prediction(session, stock, is_correct=(i % 2 == 0))

        # Mocking the prediction to avoid needing actual models
        with patch("stockai.cli.main.EnsemblePredictor") as mock_predictor, \
             patch("stockai.cli.main.YahooFinanceSource") as mock_yahoo:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock predictor
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": True, "lstm": True}
            mock_instance.predict.return_value = {
                "direction": "UP",
                "probability": 0.7,
                "confidence": 0.75,
                "confidence_level": "HIGH",
                "model_agreement": True,
                "contributions": {},
            }
            mock_predictor.return_value = mock_instance

            result = runner.invoke(app, ["predict", "TSTA", "--show-accuracy"])

        # Should show prediction and accuracy metrics
        assert result.exit_code == 0
        assert "TSTA" in result.stdout.upper()
        # Should show accuracy rate
        assert "%" in result.stdout  # Accuracy shown as percentage
        assert "Accuracy" in result.stdout

    def test_predict_show_accuracy_warns_on_low_accuracy(self):
        """Test --show-accuracy warns when accuracy is low."""
        with session_scope() as session:
            stock = self._create_test_stock(session, symbol="TSTL")
            # Create predictions with low accuracy (all wrong)
            for _ in range(10):
                self._create_evaluated_prediction(session, stock, is_correct=False)

        # Mocking the prediction to avoid needing actual models
        with patch("stockai.cli.main.EnsemblePredictor") as mock_predictor, \
             patch("stockai.cli.main.YahooFinanceSource") as mock_yahoo:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock predictor
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": True, "lstm": True}
            mock_instance.predict.return_value = {
                "direction": "DOWN",
                "probability": 0.3,
                "confidence": 0.6,
                "confidence_level": "MEDIUM",
                "model_agreement": True,
                "contributions": {},
            }
            mock_predictor.return_value = mock_instance

            result = runner.invoke(app, ["predict", "TSTL", "--show-accuracy"])

        # Should show warning for low accuracy
        assert result.exit_code == 0
        assert "TSTL" in result.stdout.upper()
        # Should warn about low accuracy (0% accuracy)
        assert "Warning" in result.stdout or "warning" in result.stdout.lower()

    def test_predict_show_accuracy_short_option(self):
        """Test -a short option works for --show-accuracy."""
        with session_scope() as session:
            self._create_test_stock(session, symbol="TSTS")

        # Mocking the prediction to avoid needing actual models
        with patch("stockai.cli.main.EnsemblePredictor") as mock_predictor, \
             patch("stockai.cli.main.YahooFinanceSource") as mock_yahoo:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock predictor
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": True, "lstm": True}
            mock_instance.predict.return_value = {
                "direction": "UP",
                "probability": 0.8,
                "confidence": 0.9,
                "confidence_level": "HIGH",
                "model_agreement": True,
                "contributions": {},
            }
            mock_predictor.return_value = mock_instance

            result = runner.invoke(app, ["predict", "TSTS", "-a"])

        # Should work with short option -a
        assert result.exit_code == 0
        assert "TSTS" in result.stdout.upper()
        assert "Accuracy" in result.stdout

    def test_predict_without_show_accuracy_no_accuracy_panel(self):
        """Test predict without --show-accuracy doesn't show accuracy panel."""
        with session_scope() as session:
            stock = self._create_test_stock(session, symbol="TSTN")
            # Create some predictions
            for _ in range(3):
                self._create_evaluated_prediction(session, stock, is_correct=True)

        # Mocking the prediction to avoid needing actual models
        with patch("stockai.cli.main.EnsemblePredictor") as mock_predictor, \
             patch("stockai.cli.main.YahooFinanceSource") as mock_yahoo:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock predictor
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": True, "lstm": True}
            mock_instance.predict.return_value = {
                "direction": "UP",
                "probability": 0.7,
                "confidence": 0.75,
                "confidence_level": "HIGH",
                "model_agreement": True,
                "contributions": {},
            }
            mock_predictor.return_value = mock_instance

            # Run without --show-accuracy
            result = runner.invoke(app, ["predict", "TSTN"])

        # Should NOT show historical accuracy panel
        assert result.exit_code == 0
        assert "TSTN" in result.stdout.upper()
        # Should not show "Historical Accuracy for" panel title
        assert "Historical Accuracy for" not in result.stdout


class TestPredictEndpointWithHistoricalAccuracy(TestPredictionAccuracyAPI):
    """Tests for GET /api/predict/{symbol} endpoint with historical accuracy."""

    def test_predict_endpoint_includes_historical_accuracy_field(self):
        """Predict endpoint response should include historical_accuracy field."""
        with session_scope() as session:
            stock = self._create_test_stock(session, symbol="BBRI")

        # Mock the EnsemblePredictor and YahooFinanceSource
        with patch("stockai.web.routes.YahooFinanceSource") as mock_yahoo, \
             patch("stockai.web.routes.EnsemblePredictor") as mock_predictor, \
             patch("stockai.web.routes.get_settings") as mock_settings:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock settings
            mock_settings.return_value.project_root = MagicMock()
            mock_settings.return_value.project_root.__truediv__ = lambda s, x: MagicMock()

            # Mock predictor
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": True, "lstm": True}
            mock_instance.predict_with_sentiment.return_value = {
                "direction": "UP",
                "probability": 0.7,
                "confidence": 0.75,
            }
            mock_predictor.return_value = mock_instance

            response = self.client.get("/api/predict/BBRI")

        assert response.status_code == 200
        data = response.json()
        assert "historical_accuracy" in data

    def test_predict_endpoint_no_historical_predictions(self):
        """Should return None for historical_accuracy when stock has no predictions."""
        with session_scope() as session:
            self._create_test_stock(session, symbol="NEWX")

        # Mock the EnsemblePredictor and YahooFinanceSource
        with patch("stockai.web.routes.YahooFinanceSource") as mock_yahoo, \
             patch("stockai.web.routes.EnsemblePredictor") as mock_predictor, \
             patch("stockai.web.routes.get_settings") as mock_settings:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock settings
            mock_settings.return_value.project_root = MagicMock()
            mock_settings.return_value.project_root.__truediv__ = lambda s, x: MagicMock()

            # Mock predictor
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": True, "lstm": True}
            mock_instance.predict_with_sentiment.return_value = {
                "direction": "UP",
                "probability": 0.7,
                "confidence": 0.75,
            }
            mock_predictor.return_value = mock_instance

            response = self.client.get("/api/predict/NEWX")

        assert response.status_code == 200
        data = response.json()
        # Should return None for stocks with no predictions
        assert data["historical_accuracy"] is None

    def test_predict_endpoint_with_historical_predictions(self):
        """Should return accuracy metrics when stock has historical predictions."""
        with session_scope() as session:
            stock = self._create_test_stock(session, symbol="ACCX")
            # 3 correct, 2 wrong = 60% accuracy
            for _ in range(3):
                self._create_evaluated_prediction(session, stock, is_correct=True)
            for _ in range(2):
                self._create_evaluated_prediction(session, stock, is_correct=False)

        # Mock the EnsemblePredictor and YahooFinanceSource
        with patch("stockai.web.routes.YahooFinanceSource") as mock_yahoo, \
             patch("stockai.web.routes.EnsemblePredictor") as mock_predictor, \
             patch("stockai.web.routes.get_settings") as mock_settings:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock settings
            mock_settings.return_value.project_root = MagicMock()
            mock_settings.return_value.project_root.__truediv__ = lambda s, x: MagicMock()

            # Mock predictor
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": True, "lstm": True}
            mock_instance.predict_with_sentiment.return_value = {
                "direction": "UP",
                "probability": 0.7,
                "confidence": 0.75,
            }
            mock_predictor.return_value = mock_instance

            response = self.client.get("/api/predict/ACCX")

        assert response.status_code == 200
        data = response.json()
        assert data["historical_accuracy"] is not None
        assert data["historical_accuracy"]["total_predictions"] == 5
        assert data["historical_accuracy"]["correct_predictions"] == 3
        assert data["historical_accuracy"]["accuracy_rate"] == 60.0

    def test_predict_endpoint_historical_accuracy_includes_direction_breakdown(self):
        """Historical accuracy should include direction breakdown."""
        with session_scope() as session:
            stock = self._create_test_stock(session, symbol="DIRX")
            self._create_evaluated_prediction(session, stock, direction="UP", is_correct=True)
            self._create_evaluated_prediction(session, stock, direction="DOWN", is_correct=False)

        # Mock the EnsemblePredictor and YahooFinanceSource
        with patch("stockai.web.routes.YahooFinanceSource") as mock_yahoo, \
             patch("stockai.web.routes.EnsemblePredictor") as mock_predictor, \
             patch("stockai.web.routes.get_settings") as mock_settings:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock settings
            mock_settings.return_value.project_root = MagicMock()
            mock_settings.return_value.project_root.__truediv__ = lambda s, x: MagicMock()

            # Mock predictor
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": True, "lstm": True}
            mock_instance.predict_with_sentiment.return_value = {
                "direction": "UP",
                "probability": 0.7,
                "confidence": 0.75,
            }
            mock_predictor.return_value = mock_instance

            response = self.client.get("/api/predict/DIRX")

        assert response.status_code == 200
        data = response.json()
        assert "by_direction" in data["historical_accuracy"]
        assert "UP" in data["historical_accuracy"]["by_direction"]
        assert "DOWN" in data["historical_accuracy"]["by_direction"]

    def test_predict_endpoint_historical_accuracy_includes_confidence_breakdown(self):
        """Historical accuracy should include confidence breakdown."""
        with session_scope() as session:
            stock = self._create_test_stock(session, symbol="CNFX")
            # HIGH confidence prediction
            self._create_evaluated_prediction(session, stock, confidence=0.8, is_correct=True)
            # LOW confidence prediction
            self._create_evaluated_prediction(session, stock, confidence=0.3, is_correct=False)

        # Mock the EnsemblePredictor and YahooFinanceSource
        with patch("stockai.web.routes.YahooFinanceSource") as mock_yahoo, \
             patch("stockai.web.routes.EnsemblePredictor") as mock_predictor, \
             patch("stockai.web.routes.get_settings") as mock_settings:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock settings
            mock_settings.return_value.project_root = MagicMock()
            mock_settings.return_value.project_root.__truediv__ = lambda s, x: MagicMock()

            # Mock predictor
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": True, "lstm": True}
            mock_instance.predict_with_sentiment.return_value = {
                "direction": "UP",
                "probability": 0.7,
                "confidence": 0.75,
            }
            mock_predictor.return_value = mock_instance

            response = self.client.get("/api/predict/CNFX")

        assert response.status_code == 200
        data = response.json()
        assert "by_confidence" in data["historical_accuracy"]
        assert "HIGH" in data["historical_accuracy"]["by_confidence"]
        assert "LOW" in data["historical_accuracy"]["by_confidence"]

    def test_predict_endpoint_no_models_includes_accuracy_field(self):
        """Response should include historical_accuracy field even when no models available."""
        with session_scope() as session:
            self._create_test_stock(session, symbol="NOMX")

        # Mock the EnsemblePredictor and YahooFinanceSource
        with patch("stockai.web.routes.YahooFinanceSource") as mock_yahoo, \
             patch("stockai.web.routes.EnsemblePredictor") as mock_predictor, \
             patch("stockai.web.routes.get_settings") as mock_settings:
            # Mock Yahoo to return enough data
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.__len__ = lambda s: 100
            mock_yahoo.return_value.get_price_history.return_value = mock_df

            # Mock settings
            mock_settings.return_value.project_root = MagicMock()
            mock_settings.return_value.project_root.__truediv__ = lambda s, x: MagicMock()

            # Mock predictor - no models loaded
            mock_instance = MagicMock()
            mock_instance.load_models.return_value = {"xgboost": False, "lstm": False}
            mock_predictor.return_value = mock_instance

            response = self.client.get("/api/predict/NOMX")

        assert response.status_code == 200
        data = response.json()
        # Should include historical_accuracy field
        assert "historical_accuracy" in data
        assert data["historical_accuracy"] is None
        assert data["prediction"] is None
        assert "No trained models" in data["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
