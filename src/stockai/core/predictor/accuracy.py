"""Prediction Accuracy Tracker for StockAI.

Tracks and evaluates prediction accuracy by filling in actual outcomes
after target dates pass.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from stockai.data.database import session_scope
from stockai.data.models import Prediction, Stock
from stockai.data.sources.yahoo import YahooFinanceSource

logger = logging.getLogger(__name__)

# Threshold for NEUTRAL direction: if absolute return is below this percentage,
# the actual direction is considered NEUTRAL
NEUTRAL_THRESHOLD_PERCENT = 0.5

# Maximum days to look back/forward for business day adjustment
MAX_BUSINESS_DAY_ADJUSTMENT = 5


def _adjust_to_business_day(date: datetime, forward: bool = True) -> datetime:
    """Adjust a date to the nearest business day.

    Handles weekends by moving to the closest trading day.
    Note: Does not handle market holidays (would need calendar data).

    Args:
        date: The date to adjust
        forward: If True, move forward to next business day;
                 if False, move backward to previous business day

    Returns:
        Adjusted datetime on a business day (weekday)
    """
    adjusted = date
    attempts = 0

    while attempts < MAX_BUSINESS_DAY_ADJUSTMENT:
        # weekday(): Monday = 0, Sunday = 6
        if adjusted.weekday() < 5:  # Monday to Friday
            return adjusted

        # Move forward or backward by 1 day
        if forward:
            adjusted = adjusted + timedelta(days=1)
        else:
            adjusted = adjusted - timedelta(days=1)
        attempts += 1

    # Return original if no business day found within limit
    return date


class PredictionAccuracyTracker:
    """Tracks and evaluates prediction accuracy.

    Features:
    - Updates past predictions with actual outcomes
    - Calculates actual direction and returns
    - Determines prediction correctness
    - Provides accuracy statistics
    """

    def __init__(self, session: Session | None = None):
        """Initialize prediction accuracy tracker.

        Args:
            session: Optional SQLAlchemy session
        """
        self._session = session
        self._use_context_manager = session is None
        self._yahoo = YahooFinanceSource()

    def get_pending_predictions(self) -> list[dict[str, Any]]:
        """Get predictions that need accuracy updates.

        Returns predictions where target_date has passed but
        is_correct is still null.

        Returns:
            List of pending prediction dictionaries
        """

        def _execute(session: Session) -> list[dict]:
            now = datetime.utcnow()

            predictions = (
                session.query(Prediction)
                .join(Stock)
                .filter(Prediction.target_date < now)
                .filter(Prediction.is_correct.is_(None))
                .order_by(Prediction.target_date.desc())
                .all()
            )

            return [
                {
                    "id": p.id,
                    "symbol": p.stock.symbol,
                    "stock_id": p.stock_id,
                    "prediction_date": p.prediction_date,
                    "target_date": p.target_date,
                    "direction": p.direction,
                    "confidence": p.confidence,
                }
                for p in predictions
            ]

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def _calculate_actual_values(
        self,
        symbol: str,
        prediction_date: datetime,
        target_date: datetime,
    ) -> dict[str, Any] | None:
        """Calculate actual direction and return for a prediction.

        Fetches historical prices and calculates the percentage return
        between prediction_date and target_date.

        Handles edge cases:
        - Missing price data: Returns None with appropriate logging
        - Weekends/holidays: Adjusts dates to nearest business day
        - Neutral case: Uses NEUTRAL_THRESHOLD_PERCENT to determine
          if movement is significant enough to be UP/DOWN

        Args:
            symbol: Stock symbol
            prediction_date: Date when prediction was made
            target_date: Target date for the prediction

        Returns:
            Dictionary with actual_direction, actual_return, or None if data unavailable
        """
        try:
            # Adjust dates to business days (prediction looks back, target looks forward)
            adjusted_pred_date = _adjust_to_business_day(prediction_date, forward=False)
            adjusted_target_date = _adjust_to_business_day(target_date, forward=True)

            # Fetch historical data covering both dates plus buffer for edge cases
            fetch_start = adjusted_pred_date - timedelta(days=7)
            fetch_end = adjusted_target_date + timedelta(days=7)

            df = self._yahoo.get_price_history(
                symbol,
                start=fetch_start,
                end=fetch_end,
            )

            if df is None or df.empty:
                logger.warning(
                    f"No price data found for {symbol} between "
                    f"{fetch_start.date()} and {fetch_end.date()}"
                )
                return None

            # Validate we have enough data
            if len(df) < 2:
                logger.warning(
                    f"Insufficient price data for {symbol}: only {len(df)} records"
                )
                return None

            # Find prices closest to prediction and target dates
            df = df.sort_values("date")
            df["date"] = df["date"].dt.normalize()

            # Normalize adjusted dates for comparison
            pred_date_normalized = adjusted_pred_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            target_date_normalized = adjusted_target_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            # Find closest price to prediction date (on or before)
            pred_prices = df[df["date"] <= pred_date_normalized]
            if pred_prices.empty:
                # No prices before pred date, use earliest available
                pred_prices = df.head(1)
                logger.info(
                    f"Using earliest available price for {symbol} prediction date"
                )
            pred_price = float(pred_prices.iloc[-1]["close"])
            actual_pred_date = pred_prices.iloc[-1]["date"]

            # Find closest price to target date (on or before)
            target_prices = df[df["date"] <= target_date_normalized]
            if target_prices.empty:
                # No prices before target date, use latest available
                target_prices = df.tail(1)
                logger.info(
                    f"Using latest available price for {symbol} target date"
                )
            target_price = float(target_prices.iloc[-1]["close"])
            actual_target_date = target_prices.iloc[-1]["date"]

            # Validate prices are different dates (sanity check)
            if actual_pred_date == actual_target_date:
                logger.warning(
                    f"Same date used for both prediction and target for {symbol}: "
                    f"{actual_pred_date}"
                )
                # Still calculate, but log the issue

            # Validate prices are positive (data quality check)
            if pred_price <= 0 or target_price <= 0:
                logger.error(
                    f"Invalid prices for {symbol}: "
                    f"pred_price={pred_price}, target_price={target_price}"
                )
                return None

            # Calculate return percentage
            actual_return = ((target_price - pred_price) / pred_price) * 100

            # Determine actual direction using NEUTRAL threshold
            # If movement is within threshold, it's considered NEUTRAL
            if abs(actual_return) < NEUTRAL_THRESHOLD_PERCENT:
                actual_direction = "NEUTRAL"
            elif actual_return > 0:
                actual_direction = "UP"
            else:
                actual_direction = "DOWN"

            return {
                "actual_direction": actual_direction,
                "actual_return": round(actual_return, 4),
                "pred_price": pred_price,
                "target_price": target_price,
                "actual_pred_date": actual_pred_date,
                "actual_target_date": actual_target_date,
            }

        except Exception as e:
            logger.error(f"Error calculating actual values for {symbol}: {e}")
            return None

    def update_past_predictions(self) -> dict[str, Any]:
        """Update all pending predictions with actual outcomes.

        Finds all predictions where target_date has passed and is_correct
        is null, then fills in actual_direction, actual_return, and is_correct.

        Returns:
            Dictionary with update statistics
        """

        def _execute(session: Session) -> dict:
            now = datetime.utcnow()

            # Get pending predictions
            predictions = (
                session.query(Prediction)
                .join(Stock)
                .filter(Prediction.target_date < now)
                .filter(Prediction.is_correct.is_(None))
                .all()
            )

            if not predictions:
                return {
                    "updated_count": 0,
                    "skipped_count": 0,
                    "error_count": 0,
                    "message": "No predictions to update",
                }

            updated_count = 0
            skipped_count = 0
            error_count = 0
            errors = []

            for prediction in predictions:
                try:
                    symbol = prediction.stock.symbol

                    # Calculate actual values
                    actual_values = self._calculate_actual_values(
                        symbol=symbol,
                        prediction_date=prediction.prediction_date,
                        target_date=prediction.target_date,
                    )

                    if actual_values is None:
                        skipped_count += 1
                        continue

                    # Update prediction
                    prediction.actual_direction = actual_values["actual_direction"]
                    prediction.actual_return = actual_values["actual_return"]

                    # Determine if prediction was correct
                    # Prediction is correct if predicted direction matches actual direction
                    predicted_direction = prediction.direction
                    actual_direction = actual_values["actual_direction"]

                    if predicted_direction == actual_direction:
                        # Exact match - prediction is correct
                        prediction.is_correct = True
                    elif predicted_direction == "NEUTRAL":
                        # We predicted NEUTRAL - correct if actual return is within threshold
                        prediction.is_correct = (
                            abs(actual_values["actual_return"]) < NEUTRAL_THRESHOLD_PERCENT
                        )
                    elif actual_direction == "NEUTRAL":
                        # Actual is NEUTRAL but we predicted UP/DOWN
                        # This means the movement was too small to matter
                        prediction.is_correct = False
                    else:
                        # Predicted UP but went DOWN, or predicted DOWN but went UP
                        prediction.is_correct = False

                    updated_count += 1

                except Exception as e:
                    error_count += 1
                    errors.append(f"{prediction.stock.symbol}: {str(e)}")
                    logger.error(
                        f"Error updating prediction {prediction.id}: {e}"
                    )

            # Commit changes
            session.commit()

            result = {
                "updated_count": updated_count,
                "skipped_count": skipped_count,
                "error_count": error_count,
                "total_pending": len(predictions),
            }

            if errors:
                result["errors"] = errors[:5]  # Limit to first 5 errors

            return result

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def get_accuracy_metrics(self) -> dict[str, Any]:
        """Get overall prediction accuracy metrics.

        Calculates accuracy statistics across all predictions that have
        been evaluated (is_correct is not null).

        Returns:
            Dictionary with accuracy metrics including:
            - total_predictions: Total number of evaluated predictions
            - correct_predictions: Number of correct predictions
            - accuracy_rate: Percentage of correct predictions (0-100)
            - by_direction: Accuracy breakdown by predicted direction (UP/DOWN/NEUTRAL)
            - by_confidence: Accuracy breakdown by confidence level (HIGH/MEDIUM/LOW)
        """

        def _execute(session: Session) -> dict:
            # Get all evaluated predictions (where is_correct is not null)
            predictions = (
                session.query(Prediction)
                .filter(Prediction.is_correct.isnot(None))
                .all()
            )

            # Handle zero predictions gracefully
            if not predictions:
                return {
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "accuracy_rate": 0.0,
                    "by_direction": {
                        "UP": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "DOWN": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "NEUTRAL": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                    },
                    "by_confidence": {
                        "HIGH": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "MEDIUM": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "LOW": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                    },
                    "message": "No evaluated predictions found",
                }

            # Calculate overall metrics
            total_predictions = len(predictions)
            correct_predictions = sum(1 for p in predictions if p.is_correct)
            accuracy_rate = (correct_predictions / total_predictions) * 100

            # Calculate accuracy by direction (UP/DOWN/NEUTRAL)
            direction_stats = {}
            for direction in ["UP", "DOWN", "NEUTRAL"]:
                direction_preds = [p for p in predictions if p.direction == direction]
                total = len(direction_preds)
                correct = sum(1 for p in direction_preds if p.is_correct)
                direction_stats[direction] = {
                    "total": total,
                    "correct": correct,
                    "accuracy_rate": round((correct / total * 100), 2) if total > 0 else 0.0,
                }

            # Calculate accuracy by confidence level
            # HIGH: confidence >= 0.7
            # MEDIUM: 0.4 <= confidence < 0.7
            # LOW: confidence < 0.4
            confidence_stats = {
                "HIGH": {"total": 0, "correct": 0},
                "MEDIUM": {"total": 0, "correct": 0},
                "LOW": {"total": 0, "correct": 0},
            }

            for p in predictions:
                confidence = p.confidence or 0
                if confidence >= 0.7:
                    level = "HIGH"
                elif confidence >= 0.4:
                    level = "MEDIUM"
                else:
                    level = "LOW"

                confidence_stats[level]["total"] += 1
                if p.is_correct:
                    confidence_stats[level]["correct"] += 1

            # Calculate accuracy rates for confidence levels
            for level, stats in confidence_stats.items():
                total = stats["total"]
                correct = stats["correct"]
                stats["accuracy_rate"] = round((correct / total * 100), 2) if total > 0 else 0.0

            return {
                "total_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "accuracy_rate": round(accuracy_rate, 2),
                "by_direction": direction_stats,
                "by_confidence": confidence_stats,
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def get_stock_accuracy(self, symbol: str) -> dict[str, Any]:
        """Get accuracy metrics for a specific stock symbol.

        Calculates stock-specific accuracy statistics and includes
        recent predictions with their outcomes, plus accuracy trend over time.

        Args:
            symbol: Stock ticker symbol (e.g., "BBRI.JK")

        Returns:
            Dictionary with stock-specific accuracy metrics including:
            - symbol: The stock symbol
            - total_predictions: Total evaluated predictions for this stock
            - correct_predictions: Number of correct predictions
            - accuracy_rate: Percentage of correct predictions (0-100)
            - by_direction: Accuracy breakdown by predicted direction
            - by_confidence: Accuracy breakdown by confidence level
            - recent_predictions: List of recent predictions with outcomes
            - accuracy_trend: Monthly accuracy trend over time
            Returns empty metrics gracefully for stocks with no predictions
        """

        def _execute(session: Session) -> dict:
            # First, find the stock by symbol
            stock = session.query(Stock).filter(Stock.symbol == symbol).first()

            # Handle stock not found
            if not stock:
                return {
                    "symbol": symbol,
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "accuracy_rate": 0.0,
                    "by_direction": {
                        "UP": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "DOWN": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "NEUTRAL": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                    },
                    "by_confidence": {
                        "HIGH": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "MEDIUM": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "LOW": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                    },
                    "recent_predictions": [],
                    "accuracy_trend": [],
                    "message": f"Stock '{symbol}' not found",
                }

            # Get all evaluated predictions for this stock (where is_correct is not null)
            predictions = (
                session.query(Prediction)
                .filter(Prediction.stock_id == stock.id)
                .filter(Prediction.is_correct.isnot(None))
                .order_by(Prediction.target_date.desc())
                .all()
            )

            # Handle no predictions gracefully
            if not predictions:
                return {
                    "symbol": symbol,
                    "stock_name": stock.name,
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "accuracy_rate": 0.0,
                    "by_direction": {
                        "UP": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "DOWN": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "NEUTRAL": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                    },
                    "by_confidence": {
                        "HIGH": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "MEDIUM": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                        "LOW": {"total": 0, "correct": 0, "accuracy_rate": 0.0},
                    },
                    "recent_predictions": [],
                    "accuracy_trend": [],
                    "message": f"No evaluated predictions for '{symbol}'",
                }

            # Calculate overall metrics
            total_predictions = len(predictions)
            correct_predictions = sum(1 for p in predictions if p.is_correct)
            accuracy_rate = (correct_predictions / total_predictions) * 100

            # Calculate accuracy by direction (UP/DOWN/NEUTRAL)
            direction_stats = {}
            for direction in ["UP", "DOWN", "NEUTRAL"]:
                direction_preds = [p for p in predictions if p.direction == direction]
                total = len(direction_preds)
                correct = sum(1 for p in direction_preds if p.is_correct)
                direction_stats[direction] = {
                    "total": total,
                    "correct": correct,
                    "accuracy_rate": round((correct / total * 100), 2) if total > 0 else 0.0,
                }

            # Calculate accuracy by confidence level
            confidence_stats = {
                "HIGH": {"total": 0, "correct": 0},
                "MEDIUM": {"total": 0, "correct": 0},
                "LOW": {"total": 0, "correct": 0},
            }

            for p in predictions:
                confidence = p.confidence or 0
                if confidence >= 0.7:
                    level = "HIGH"
                elif confidence >= 0.4:
                    level = "MEDIUM"
                else:
                    level = "LOW"

                confidence_stats[level]["total"] += 1
                if p.is_correct:
                    confidence_stats[level]["correct"] += 1

            # Calculate accuracy rates for confidence levels
            for level, stats in confidence_stats.items():
                total = stats["total"]
                correct = stats["correct"]
                stats["accuracy_rate"] = round((correct / total * 100), 2) if total > 0 else 0.0

            # Get recent predictions with outcomes (last 10)
            recent_predictions = []
            for p in predictions[:10]:
                recent_predictions.append({
                    "id": p.id,
                    "prediction_date": p.prediction_date.isoformat() if p.prediction_date else None,
                    "target_date": p.target_date.isoformat() if p.target_date else None,
                    "direction": p.direction,
                    "confidence": round(p.confidence, 4) if p.confidence else None,
                    "actual_direction": p.actual_direction,
                    "actual_return": round(p.actual_return, 4) if p.actual_return else None,
                    "is_correct": p.is_correct,
                })

            # Calculate accuracy trend over time (by month)
            # Group predictions by month and calculate accuracy for each
            monthly_accuracy = {}
            for p in predictions:
                if p.target_date:
                    # Use target_date for the month key
                    month_key = p.target_date.strftime("%Y-%m")
                    if month_key not in monthly_accuracy:
                        monthly_accuracy[month_key] = {"total": 0, "correct": 0}
                    monthly_accuracy[month_key]["total"] += 1
                    if p.is_correct:
                        monthly_accuracy[month_key]["correct"] += 1

            # Convert to sorted list of accuracy trend
            accuracy_trend = []
            for month in sorted(monthly_accuracy.keys()):
                stats = monthly_accuracy[month]
                accuracy_trend.append({
                    "month": month,
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy_rate": round(
                        (stats["correct"] / stats["total"] * 100), 2
                    ) if stats["total"] > 0 else 0.0,
                })

            return {
                "symbol": symbol,
                "stock_name": stock.name,
                "total_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "accuracy_rate": round(accuracy_rate, 2),
                "by_direction": direction_stats,
                "by_confidence": confidence_stats,
                "recent_predictions": recent_predictions,
                "accuracy_trend": accuracy_trend,
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)

    def get_accuracy_by_model(self) -> dict[str, Any]:
        """Analyze which model components correlate with correct predictions.

        Examines the relationship between individual model component probabilities
        (XGBoost, LSTM, sentiment) and prediction correctness to identify which
        components are most predictive.

        Returns:
            Dictionary with model-specific accuracy analysis including:
            - xgboost: Accuracy metrics binned by xgboost_prob ranges
            - lstm: Accuracy metrics binned by lstm_prob ranges
            - sentiment: Accuracy metrics binned by sentiment_score ranges
            - insights: List of observations about model performance
            - correlation_summary: Overall correlation indicators for each component
        """

        def _execute(session: Session) -> dict:
            # Get all evaluated predictions with model component data
            predictions = (
                session.query(Prediction)
                .filter(Prediction.is_correct.isnot(None))
                .all()
            )

            # Handle zero predictions gracefully
            if not predictions:
                return {
                    "xgboost": {"bins": [], "total_with_data": 0},
                    "lstm": {"bins": [], "total_with_data": 0},
                    "sentiment": {"bins": [], "total_with_data": 0},
                    "correlation_summary": {
                        "xgboost": {"has_correlation": None, "direction": None},
                        "lstm": {"has_correlation": None, "direction": None},
                        "sentiment": {"has_correlation": None, "direction": None},
                    },
                    "insights": [],
                    "message": "No evaluated predictions found",
                }

            # Define probability bins for analysis
            # For xgboost_prob and lstm_prob: 0-1 scale
            prob_bins = [
                {"label": "Very Low (0-0.2)", "min": 0.0, "max": 0.2},
                {"label": "Low (0.2-0.4)", "min": 0.2, "max": 0.4},
                {"label": "Medium (0.4-0.6)", "min": 0.4, "max": 0.6},
                {"label": "High (0.6-0.8)", "min": 0.6, "max": 0.8},
                {"label": "Very High (0.8-1.0)", "min": 0.8, "max": 1.0},
            ]

            # For sentiment_score: -1 to 1 scale
            sentiment_bins = [
                {"label": "Very Negative (-1.0 to -0.6)", "min": -1.0, "max": -0.6},
                {"label": "Negative (-0.6 to -0.2)", "min": -0.6, "max": -0.2},
                {"label": "Neutral (-0.2 to 0.2)", "min": -0.2, "max": 0.2},
                {"label": "Positive (0.2 to 0.6)", "min": 0.2, "max": 0.6},
                {"label": "Very Positive (0.6 to 1.0)", "min": 0.6, "max": 1.0},
            ]

            def analyze_component(
                predictions: list,
                attr_name: str,
                bins: list[dict],
            ) -> dict:
                """Analyze accuracy by component probability bins."""
                # Filter predictions that have this component's data
                preds_with_data = [
                    p for p in predictions
                    if getattr(p, attr_name) is not None
                ]

                if not preds_with_data:
                    return {
                        "bins": [],
                        "total_with_data": 0,
                        "overall_accuracy": 0.0,
                    }

                bin_results = []
                for bin_def in bins:
                    # Get predictions in this bin
                    bin_preds = [
                        p for p in preds_with_data
                        if bin_def["min"] <= getattr(p, attr_name) < bin_def["max"]
                        or (bin_def["max"] == 1.0 and getattr(p, attr_name) == 1.0)
                        or (bin_def["max"] == -0.6 and getattr(p, attr_name) == -1.0)
                    ]

                    total = len(bin_preds)
                    correct = sum(1 for p in bin_preds if p.is_correct)
                    accuracy = round((correct / total * 100), 2) if total > 0 else 0.0

                    bin_results.append({
                        "label": bin_def["label"],
                        "min": bin_def["min"],
                        "max": bin_def["max"],
                        "total": total,
                        "correct": correct,
                        "accuracy_rate": accuracy,
                    })

                # Calculate overall accuracy for this component
                total_with_data = len(preds_with_data)
                correct_with_data = sum(1 for p in preds_with_data if p.is_correct)
                overall_accuracy = round(
                    (correct_with_data / total_with_data * 100), 2
                ) if total_with_data > 0 else 0.0

                return {
                    "bins": bin_results,
                    "total_with_data": total_with_data,
                    "overall_accuracy": overall_accuracy,
                }

            # Analyze each model component
            xgboost_analysis = analyze_component(predictions, "xgboost_prob", prob_bins)
            lstm_analysis = analyze_component(predictions, "lstm_prob", prob_bins)
            sentiment_analysis = analyze_component(
                predictions, "sentiment_score", sentiment_bins
            )

            def calculate_correlation_indicator(analysis: dict) -> dict:
                """Calculate a simple correlation indicator.

                Compares accuracy in high vs low probability bins to determine
                if higher component values correlate with correct predictions.
                """
                bins = analysis.get("bins", [])
                if not bins or analysis.get("total_with_data", 0) < 5:
                    return {"has_correlation": None, "direction": None}

                # Get accuracy for low bins (first 2) vs high bins (last 2)
                low_bins = [b for b in bins[:2] if b["total"] > 0]
                high_bins = [b for b in bins[-2:] if b["total"] > 0]

                if not low_bins or not high_bins:
                    return {"has_correlation": None, "direction": None}

                # Weighted average accuracy for low and high
                low_total = sum(b["total"] for b in low_bins)
                low_correct = sum(b["correct"] for b in low_bins)
                low_accuracy = (low_correct / low_total * 100) if low_total > 0 else 0

                high_total = sum(b["total"] for b in high_bins)
                high_correct = sum(b["correct"] for b in high_bins)
                high_accuracy = (high_correct / high_total * 100) if high_total > 0 else 0

                # Determine correlation direction and strength
                diff = high_accuracy - low_accuracy
                if abs(diff) < 5:  # Less than 5% difference is weak
                    return {
                        "has_correlation": False,
                        "direction": "none",
                        "low_accuracy": round(low_accuracy, 2),
                        "high_accuracy": round(high_accuracy, 2),
                        "difference": round(diff, 2),
                    }
                elif diff > 0:
                    return {
                        "has_correlation": True,
                        "direction": "positive",
                        "low_accuracy": round(low_accuracy, 2),
                        "high_accuracy": round(high_accuracy, 2),
                        "difference": round(diff, 2),
                    }
                else:
                    return {
                        "has_correlation": True,
                        "direction": "negative",
                        "low_accuracy": round(low_accuracy, 2),
                        "high_accuracy": round(high_accuracy, 2),
                        "difference": round(diff, 2),
                    }

            # Calculate correlation indicators
            xgboost_corr = calculate_correlation_indicator(xgboost_analysis)
            lstm_corr = calculate_correlation_indicator(lstm_analysis)
            sentiment_corr = calculate_correlation_indicator(sentiment_analysis)

            # Generate insights based on analysis
            insights = []

            # XGBoost insights
            if xgboost_corr.get("has_correlation") is True:
                if xgboost_corr["direction"] == "positive":
                    insights.append(
                        f"XGBoost shows positive correlation: higher probabilities "
                        f"correlate with more correct predictions "
                        f"({xgboost_corr['high_accuracy']:.1f}% vs {xgboost_corr['low_accuracy']:.1f}%)"
                    )
                else:
                    insights.append(
                        f"XGBoost shows unexpected negative correlation: lower probabilities "
                        f"correlate with more correct predictions "
                        f"({xgboost_corr['low_accuracy']:.1f}% vs {xgboost_corr['high_accuracy']:.1f}%)"
                    )
            elif xgboost_corr.get("has_correlation") is False:
                insights.append(
                    "XGBoost probability shows weak correlation with prediction accuracy"
                )

            # LSTM insights
            if lstm_corr.get("has_correlation") is True:
                if lstm_corr["direction"] == "positive":
                    insights.append(
                        f"LSTM shows positive correlation: higher probabilities "
                        f"correlate with more correct predictions "
                        f"({lstm_corr['high_accuracy']:.1f}% vs {lstm_corr['low_accuracy']:.1f}%)"
                    )
                else:
                    insights.append(
                        f"LSTM shows unexpected negative correlation: lower probabilities "
                        f"correlate with more correct predictions "
                        f"({lstm_corr['low_accuracy']:.1f}% vs {lstm_corr['high_accuracy']:.1f}%)"
                    )
            elif lstm_corr.get("has_correlation") is False:
                insights.append(
                    "LSTM probability shows weak correlation with prediction accuracy"
                )

            # Sentiment insights
            if sentiment_corr.get("has_correlation") is True:
                if sentiment_corr["direction"] == "positive":
                    insights.append(
                        f"Sentiment score shows positive correlation: higher scores "
                        f"correlate with more correct predictions "
                        f"({sentiment_corr['high_accuracy']:.1f}% vs {sentiment_corr['low_accuracy']:.1f}%)"
                    )
                else:
                    insights.append(
                        f"Sentiment score shows negative correlation: lower scores "
                        f"correlate with more correct predictions "
                        f"({sentiment_corr['low_accuracy']:.1f}% vs {sentiment_corr['high_accuracy']:.1f}%)"
                    )
            elif sentiment_corr.get("has_correlation") is False:
                insights.append(
                    "Sentiment score shows weak correlation with prediction accuracy"
                )

            # Find best performing component
            components_with_corr = []
            if xgboost_corr.get("has_correlation") is True:
                components_with_corr.append(
                    ("XGBoost", abs(xgboost_corr.get("difference", 0)))
                )
            if lstm_corr.get("has_correlation") is True:
                components_with_corr.append(
                    ("LSTM", abs(lstm_corr.get("difference", 0)))
                )
            if sentiment_corr.get("has_correlation") is True:
                components_with_corr.append(
                    ("Sentiment", abs(sentiment_corr.get("difference", 0)))
                )

            if components_with_corr:
                best_component = max(components_with_corr, key=lambda x: x[1])
                insights.append(
                    f"Strongest predictor: {best_component[0]} "
                    f"(accuracy difference: {best_component[1]:.1f}%)"
                )

            # Check for high-confidence bins
            for name, analysis in [
                ("XGBoost", xgboost_analysis),
                ("LSTM", lstm_analysis),
            ]:
                high_bins = [b for b in analysis.get("bins", [])[-2:] if b["total"] >= 3]
                for bin_data in high_bins:
                    if bin_data["accuracy_rate"] >= 70:
                        insights.append(
                            f"{name} {bin_data['label']}: {bin_data['accuracy_rate']:.1f}% accuracy "
                            f"({bin_data['correct']}/{bin_data['total']} correct)"
                        )

            return {
                "xgboost": xgboost_analysis,
                "lstm": lstm_analysis,
                "sentiment": sentiment_analysis,
                "correlation_summary": {
                    "xgboost": xgboost_corr,
                    "lstm": lstm_corr,
                    "sentiment": sentiment_corr,
                },
                "insights": insights,
                "total_evaluated_predictions": len(predictions),
            }

        if self._use_context_manager:
            with session_scope() as session:
                return _execute(session)
        else:
            return _execute(self._session)
