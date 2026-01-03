"""Multi-Factor Scoring System.

Implements hedge fund-style quantitative scoring for Indonesian stocks.
Factors: Value (25%), Quality (30%), Momentum (25%), Volatility (20%)
"""

from stockai.scoring.factors import (
    calculate_value_score,
    calculate_quality_score,
    calculate_momentum_score,
    calculate_volatility_score,
    calculate_composite_score,
    FactorScores,
    FACTOR_WEIGHTS,
)
from stockai.scoring.screener import StockScreener, ScreeningCriteria
from stockai.scoring.signals import SignalGenerator, Signal, SignalType

__all__ = [
    "calculate_value_score",
    "calculate_quality_score",
    "calculate_momentum_score",
    "calculate_volatility_score",
    "calculate_composite_score",
    "FactorScores",
    "FACTOR_WEIGHTS",
    "StockScreener",
    "ScreeningCriteria",
    "SignalGenerator",
    "Signal",
    "SignalType",
]
