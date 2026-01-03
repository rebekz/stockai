"""Signal Generation Module.

Generates actionable BUY/SELL/HOLD signals based on:
- Composite score thresholds
- Score changes over time
- Technical confirmations
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
import pytz

import logging

logger = logging.getLogger(__name__)

TIMEZONE = pytz.timezone("Asia/Jakarta")


class SignalType(Enum):
    """Trading signal types."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalReason(Enum):
    """Reasons for signal generation."""

    SCORE_EXCELLENT = "Score >= 80 (Excellent)"
    SCORE_GOOD = "Score 70-79 (Good)"
    SCORE_FAIR = "Score 60-69 (Fair)"
    SCORE_POOR = "Score < 50 (Poor)"
    SCORE_IMPROVING = "Score increased significantly"
    SCORE_DECLINING = "Score decreased significantly"
    MOMENTUM_POSITIVE = "Strong upward momentum"
    MOMENTUM_NEGATIVE = "Weak or negative momentum"
    STOP_LOSS_HIT = "Price below stop-loss"
    TARGET_REACHED = "Price reached target"
    OVERBOUGHT = "Technical overbought condition"
    OVERSOLD = "Technical oversold condition"


@dataclass
class Signal:
    """A trading signal with context."""

    symbol: str
    signal_type: SignalType
    confidence: float  # 0-100
    reasons: list[SignalReason]
    current_score: float
    previous_score: float | None
    current_price: float
    stop_loss_suggested: float | None
    target_suggested: float | None
    timestamp: datetime

    @property
    def is_actionable(self) -> bool:
        """Check if signal requires action (not HOLD)."""
        return self.signal_type in [
            SignalType.STRONG_BUY,
            SignalType.BUY,
            SignalType.SELL,
            SignalType.STRONG_SELL,
        ]

    @property
    def risk_reward_ratio(self) -> float | None:
        """Calculate risk/reward ratio if targets set."""
        if self.stop_loss_suggested and self.target_suggested:
            risk = self.current_price - self.stop_loss_suggested
            reward = self.target_suggested - self.current_price
            if risk > 0:
                return round(reward / risk, 2)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal": self.signal_type.value,
            "confidence": round(self.confidence, 1),
            "reasons": [r.value for r in self.reasons],
            "current_score": round(self.current_score, 1),
            "previous_score": round(self.previous_score, 1) if self.previous_score else None,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss_suggested,
            "target": self.target_suggested,
            "risk_reward": self.risk_reward_ratio,
            "timestamp": self.timestamp.isoformat(),
        }


class SignalGenerator:
    """Generate trading signals from scores and prices."""

    # Score thresholds
    EXCELLENT_THRESHOLD = 80
    GOOD_THRESHOLD = 70
    FAIR_THRESHOLD = 60
    POOR_THRESHOLD = 50

    # Score change thresholds
    SIGNIFICANT_INCREASE = 10  # Points
    SIGNIFICANT_DECREASE = 10

    # Default risk parameters
    DEFAULT_STOP_LOSS_PCT = 0.08  # 8% below entry
    DEFAULT_TARGET_PCT = 0.15  # 15% above entry

    def __init__(
        self,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        target_pct: float = DEFAULT_TARGET_PCT,
    ):
        """Initialize signal generator.

        Args:
            stop_loss_pct: Default stop-loss percentage
            target_pct: Default target percentage
        """
        self.stop_loss_pct = stop_loss_pct
        self.target_pct = target_pct

    def generate_signal(
        self,
        symbol: str,
        current_score: float,
        current_price: float,
        previous_score: float | None = None,
        momentum_score: float | None = None,
        rsi: float | None = None,
    ) -> Signal:
        """Generate trading signal for a stock.

        Args:
            symbol: Stock symbol
            current_score: Current composite score
            current_price: Current stock price
            previous_score: Previous composite score (for trend)
            momentum_score: Momentum factor score
            rsi: RSI indicator value

        Returns:
            Signal with recommendation
        """
        reasons = []
        confidence_adjustments = []

        # 1. Score-based signal determination
        base_signal, base_reason = self._score_to_signal(current_score)
        reasons.append(base_reason)

        # 2. Score trend adjustment
        if previous_score is not None:
            score_change = current_score - previous_score

            if score_change >= self.SIGNIFICANT_INCREASE:
                if base_signal in [SignalType.HOLD, SignalType.BUY]:
                    base_signal = SignalType.BUY
                    confidence_adjustments.append(10)
                reasons.append(SignalReason.SCORE_IMPROVING)

            elif score_change <= -self.SIGNIFICANT_DECREASE:
                if base_signal in [SignalType.HOLD, SignalType.SELL]:
                    base_signal = SignalType.SELL
                    confidence_adjustments.append(-10)
                reasons.append(SignalReason.SCORE_DECLINING)

        # 3. Momentum confirmation
        if momentum_score is not None:
            if momentum_score >= 70 and base_signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                confidence_adjustments.append(10)
                reasons.append(SignalReason.MOMENTUM_POSITIVE)
            elif momentum_score <= 30:
                confidence_adjustments.append(-10)
                reasons.append(SignalReason.MOMENTUM_NEGATIVE)

        # 4. RSI overbought/oversold
        if rsi is not None:
            if rsi >= 70:
                reasons.append(SignalReason.OVERBOUGHT)
                if base_signal == SignalType.BUY:
                    base_signal = SignalType.HOLD
                    confidence_adjustments.append(-15)
            elif rsi <= 30:
                reasons.append(SignalReason.OVERSOLD)
                if base_signal == SignalType.SELL:
                    base_signal = SignalType.HOLD
                    confidence_adjustments.append(-15)

        # Calculate final confidence
        base_confidence = self._score_to_confidence(current_score)
        final_confidence = max(0, min(100, base_confidence + sum(confidence_adjustments)))

        # Calculate suggested stop-loss and target
        stop_loss = round(current_price * (1 - self.stop_loss_pct), 0) if base_signal in [
            SignalType.BUY,
            SignalType.STRONG_BUY,
        ] else None

        target = round(current_price * (1 + self.target_pct), 0) if base_signal in [
            SignalType.BUY,
            SignalType.STRONG_BUY,
        ] else None

        return Signal(
            symbol=symbol,
            signal_type=base_signal,
            confidence=final_confidence,
            reasons=reasons,
            current_score=current_score,
            previous_score=previous_score,
            current_price=current_price,
            stop_loss_suggested=stop_loss,
            target_suggested=target,
            timestamp=datetime.now(TIMEZONE),
        )

    def _score_to_signal(self, score: float) -> tuple[SignalType, SignalReason]:
        """Convert composite score to base signal.

        Args:
            score: Composite score 0-100

        Returns:
            Tuple of (SignalType, SignalReason)
        """
        if score >= self.EXCELLENT_THRESHOLD:
            return SignalType.STRONG_BUY, SignalReason.SCORE_EXCELLENT
        elif score >= self.GOOD_THRESHOLD:
            return SignalType.BUY, SignalReason.SCORE_GOOD
        elif score >= self.FAIR_THRESHOLD:
            return SignalType.HOLD, SignalReason.SCORE_FAIR
        elif score >= self.POOR_THRESHOLD:
            return SignalType.HOLD, SignalReason.SCORE_FAIR
        else:
            return SignalType.SELL, SignalReason.SCORE_POOR

    def _score_to_confidence(self, score: float) -> float:
        """Convert score to base confidence level.

        Args:
            score: Composite score 0-100

        Returns:
            Base confidence 0-100
        """
        # Higher scores = higher confidence
        if score >= 80:
            return 85
        elif score >= 70:
            return 70
        elif score >= 60:
            return 55
        elif score >= 50:
            return 40
        else:
            return 70  # High confidence in sell signal for bad stocks

    def check_stop_loss(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> Signal | None:
        """Check if stop-loss is triggered.

        Args:
            symbol: Stock symbol
            current_price: Current price
            entry_price: Original entry price
            stop_loss_price: Stop-loss price

        Returns:
            SELL Signal if triggered, None otherwise
        """
        if current_price <= stop_loss_price:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.STRONG_SELL,
                confidence=95,
                reasons=[SignalReason.STOP_LOSS_HIT],
                current_score=0,  # N/A for stop-loss
                previous_score=None,
                current_price=current_price,
                stop_loss_suggested=None,
                target_suggested=None,
                timestamp=datetime.now(TIMEZONE),
            )
        return None

    def check_target(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        target_price: float,
    ) -> Signal | None:
        """Check if target is reached.

        Args:
            symbol: Stock symbol
            current_price: Current price
            entry_price: Original entry price
            target_price: Target price

        Returns:
            SELL Signal if target reached, None otherwise
        """
        if current_price >= target_price:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=80,
                reasons=[SignalReason.TARGET_REACHED],
                current_score=0,  # N/A for target
                previous_score=None,
                current_price=current_price,
                stop_loss_suggested=None,
                target_suggested=None,
                timestamp=datetime.now(TIMEZONE),
            )
        return None


def format_signal_for_display(signal: Signal) -> str:
    """Format signal for CLI display.

    Args:
        signal: Signal to format

    Returns:
        Formatted string
    """
    emoji = {
        SignalType.STRONG_BUY: "🟢🟢",
        SignalType.BUY: "🟢",
        SignalType.HOLD: "🟡",
        SignalType.SELL: "🔴",
        SignalType.STRONG_SELL: "🔴🔴",
    }

    lines = [
        f"{emoji[signal.signal_type]} {signal.symbol}: {signal.signal_type.value}",
        f"   Score: {signal.current_score:.0f}/100 | Confidence: {signal.confidence:.0f}%",
        f"   Price: Rp {signal.current_price:,.0f}",
    ]

    if signal.stop_loss_suggested:
        lines.append(f"   Stop-Loss: Rp {signal.stop_loss_suggested:,.0f}")

    if signal.target_suggested:
        lines.append(f"   Target: Rp {signal.target_suggested:,.0f}")

    if signal.risk_reward_ratio:
        lines.append(f"   Risk/Reward: 1:{signal.risk_reward_ratio}")

    lines.append(f"   Reasons: {', '.join(r.value for r in signal.reasons)}")

    return "\n".join(lines)
