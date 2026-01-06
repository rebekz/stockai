"""Integrated Stock Analyzer Service.

Main integration point combining all scoring components:
- Composite scoring (Value, Quality, Momentum, Volatility)
- Smart Money Score (OBV, MFI, Volume analysis)
- Support/Resistance detection
- ADX trend strength
- Gate validation
- Trade plan generation
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stockai.scoring.factors import (
    FactorScores,
    calculate_composite_score,
    calculate_value_score,
    calculate_quality_score,
    calculate_momentum_score,
    calculate_volatility_score,
)
from stockai.scoring.smart_money import SmartMoneyResult, calculate_smart_money_score
from stockai.scoring.support_resistance import (
    SupportResistanceResult,
    find_support_resistance,
)
from stockai.scoring.gates import GateConfig, GateResult, validate_gates
from stockai.scoring.trade_plan import TradePlan, generate_trade_plan
from stockai.tools.stock_tools import calculate_adx


@dataclass
class AnalysisResult:
    """Complete stock analysis result."""

    ticker: str
    current_price: float

    # Existing scores (from factors.py)
    composite_score: float
    value_score: float
    quality_score: float
    momentum_score: float
    volatility_score: float

    # New scores
    smart_money: SmartMoneyResult
    support_resistance: SupportResistanceResult
    adx: dict[str, Any]

    # Gate validation
    gates: GateResult

    # Trade plan (if qualified)
    trade_plan: TradePlan | None

    # Final decision
    decision: str  # BUY, NO_TRADE
    confidence: str  # HIGH, WATCH, REJECTED


def analyze_stock(
    ticker: str,
    df: pd.DataFrame,
    fundamentals: dict[str, Any] | None = None,
    config: GateConfig | None = None,
) -> AnalysisResult:
    """Perform complete stock analysis with gate validation.

    Args:
        ticker: Stock ticker symbol
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        fundamentals: Optional dictionary with fundamental data
            (pe_ratio, pb_ratio, roe, debt_to_equity, profit_margin, current_ratio)
        config: Optional GateConfig for custom thresholds

    Returns:
        AnalysisResult with complete analysis and trade plan if qualified
    """
    if config is None:
        config = GateConfig()

    fundamentals = fundamentals or {}

    # Get current price
    current_price = df["close"].iloc[-1]

    # Calculate composite scores
    factor_scores = _calculate_factor_scores(df, fundamentals)

    # Calculate Smart Money Score
    smart_money = calculate_smart_money_score(df)

    # Find Support/Resistance
    support_resistance = find_support_resistance(df)

    # Calculate ADX
    adx = calculate_adx(df)

    # Prepare gate validation data
    # Map existing factor scores to gate requirements
    # Technical score: average of momentum and volatility (inverted)
    technical_score = (factor_scores.momentum_score + (100 - factor_scores.volatility_score)) / 2
    # Fundamental score: average of value and quality
    fundamental_score = (factor_scores.value_score + factor_scores.quality_score) / 2

    gate_data = {
        "overall_score": factor_scores.composite_score,
        "technical_score": technical_score,
        "smart_money_score": smart_money.score,
        "distance_to_support_pct": support_resistance.distance_to_support_pct,
        "adx": adx.get("adx", 0),
        "fundamental_score": fundamental_score,
    }

    # Validate gates
    gates = validate_gates(gate_data, config)

    # Generate trade plan if qualified
    trade_plan = None
    if gates.all_passed or gates.confidence == "WATCH":
        trade_plan = generate_trade_plan(
            current_price=current_price,
            support=support_resistance.nearest_support,
            resistances=support_resistance.resistances,
        )

    # Determine final decision
    if gates.all_passed:
        decision = "BUY"
        confidence = "HIGH"
    elif gates.confidence == "WATCH":
        decision = "NO_TRADE"
        confidence = "WATCH"
    else:
        decision = "NO_TRADE"
        confidence = "REJECTED"

    return AnalysisResult(
        ticker=ticker.upper(),
        current_price=float(current_price),
        composite_score=factor_scores.composite_score,
        value_score=factor_scores.value_score,
        quality_score=factor_scores.quality_score,
        momentum_score=factor_scores.momentum_score,
        volatility_score=factor_scores.volatility_score,
        smart_money=smart_money,
        support_resistance=support_resistance,
        adx=adx,
        gates=gates,
        trade_plan=trade_plan,
        decision=decision,
        confidence=confidence,
    )


def _calculate_factor_scores(
    df: pd.DataFrame,
    fundamentals: dict[str, Any],
) -> FactorScores:
    """Calculate factor scores from price and fundamental data.

    This is a simplified version that works with DataFrame directly.
    For full scoring, use the factors module with complete data.
    """
    # Calculate price-based metrics
    close = df["close"]
    returns_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
    returns_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0
    returns_6m = (close.iloc[-1] / close.iloc[-126] - 1) * 100 if len(close) >= 126 else 0

    # Calculate volatility (annualized standard deviation)
    daily_returns = close.pct_change().dropna()
    volatility = daily_returns.std() * (252 ** 0.5) * 100 if len(daily_returns) > 0 else 30

    # Use defaults for missing fundamentals
    pe_ratio = fundamentals.get("pe_ratio")
    pb_ratio = fundamentals.get("pb_ratio")
    roe = fundamentals.get("roe")
    debt_to_equity = fundamentals.get("debt_to_equity")
    profit_margin = fundamentals.get("profit_margin")
    current_ratio = fundamentals.get("current_ratio")

    # Calculate individual factor scores
    value_score = calculate_value_score(pe_ratio, pb_ratio)
    quality_score = calculate_quality_score(roe, debt_to_equity, profit_margin, current_ratio)
    momentum_score = calculate_momentum_score(returns_6m, returns_3m, returns_1m)
    volatility_score = calculate_volatility_score(beta=1.0, std_dev=volatility)

    # Calculate composite score
    composite = calculate_composite_score(
        value_score, quality_score, momentum_score, volatility_score
    )

    return FactorScores(
        symbol="",  # Will be set by caller
        value_score=value_score,
        quality_score=quality_score,
        momentum_score=momentum_score,
        volatility_score=volatility_score,
        composite_score=composite,
        pe_ratio=pe_ratio,
        pb_ratio=pb_ratio,
        roe=roe,
        debt_to_equity=debt_to_equity,
        profit_margin=profit_margin,
        momentum_6m=returns_6m,
        beta=1.0,
        std_dev=volatility,
    )
