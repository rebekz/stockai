"""Prediction Accuracy Tracker for StockAI.

Tracks and evaluates prediction accuracy by filling in actual outcomes
after target dates pass.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from stockai.data.database import session_scope
from stockai.data.models import Prediction, Stock
from stockai.data.sources.yahoo import YahooFinanceSource

logger = logging.getLogger(__name__)


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

        Args:
            symbol: Stock symbol
            prediction_date: Date when prediction was made
            target_date: Target date for the prediction

        Returns:
            Dictionary with actual_direction, actual_return, or None if data unavailable
        """
        try:
            # Fetch historical data covering both dates
            df = self._yahoo.get_price_history(
                symbol,
                start=prediction_date,
                end=target_date + __import__("datetime").timedelta(days=5),
            )

            if df is None or df.empty:
                logger.warning(f"No price data found for {symbol}")
                return None

            # Find prices closest to prediction and target dates
            df = df.sort_values("date")
            df["date"] = df["date"].dt.normalize()

            # Get prediction date price (or closest available)
            pred_date_normalized = prediction_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            target_date_normalized = target_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            # Find closest price to prediction date
            pred_prices = df[df["date"] <= pred_date_normalized]
            if pred_prices.empty:
                pred_prices = df.head(1)
            pred_price = float(pred_prices.iloc[-1]["close"])

            # Find closest price to target date
            target_prices = df[df["date"] <= target_date_normalized]
            if target_prices.empty:
                target_prices = df.tail(1)
            target_price = float(target_prices.iloc[-1]["close"])

            # Calculate return percentage
            actual_return = ((target_price - pred_price) / pred_price) * 100

            # Determine actual direction
            if actual_return > 0:
                actual_direction = "UP"
            elif actual_return < 0:
                actual_direction = "DOWN"
            else:
                actual_direction = "NEUTRAL"

            return {
                "actual_direction": actual_direction,
                "actual_return": round(actual_return, 4),
                "pred_price": pred_price,
                "target_price": target_price,
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
                    # For NEUTRAL predictions, compare with actual direction
                    predicted_direction = prediction.direction
                    actual_direction = actual_values["actual_direction"]

                    if predicted_direction == actual_direction:
                        prediction.is_correct = True
                    elif predicted_direction == "NEUTRAL":
                        # NEUTRAL is considered correct if actual return is small
                        prediction.is_correct = abs(actual_values["actual_return"]) < 0.5
                    elif actual_direction == "NEUTRAL":
                        # Actual is NEUTRAL but we predicted UP/DOWN
                        prediction.is_correct = False
                    else:
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
