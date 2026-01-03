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
