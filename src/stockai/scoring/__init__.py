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
from stockai.scoring.smart_money import SmartMoneyResult, calculate_smart_money_score
from stockai.scoring.support_resistance import (
    SupportResistanceResult,
    find_support_resistance,
)
from stockai.scoring.gates import GateConfig, GateResult, validate_gates
from stockai.scoring.trade_plan import (
    TradePlanConfig,
    TradePlan,
    generate_trade_plan,
    calculate_position_with_plan,
)
from stockai.scoring.analyzer import AnalysisResult, analyze_stock

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
    "SmartMoneyResult",
    "calculate_smart_money_score",
    "SupportResistanceResult",
    "find_support_resistance",
    "GateConfig",
    "GateResult",
    "validate_gates",
    "TradePlanConfig",
    "TradePlan",
    "generate_trade_plan",
    "calculate_position_with_plan",
    "AnalysisResult",
    "analyze_stock",
]
