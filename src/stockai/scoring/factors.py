"""Multi-Factor Scoring Module.

Implements academic factor investing approach:
- Value (25%): P/E, P/B relative to sector
- Quality (30%): ROE, debt ratios, profit margins
- Momentum (25%): 6-month price performance
- Volatility (20%): Beta, standard deviation
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

import logging

logger = logging.getLogger(__name__)

# Factor weights (hedge fund style, emphasizing Quality for beginners)
FACTOR_WEIGHTS = {
    "value": 0.25,
    "quality": 0.30,
    "momentum": 0.25,
    "volatility": 0.20,
}


@dataclass
class FactorScores:
    """Individual factor scores for a stock."""

    symbol: str
    value_score: float
    quality_score: float
    momentum_score: float
    volatility_score: float
    composite_score: float

    # Raw metrics for transparency
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    roe: float | None = None
    debt_to_equity: float | None = None
    profit_margin: float | None = None
    momentum_6m: float | None = None
    beta: float | None = None
    std_dev: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "composite_score": round(self.composite_score, 1),
            "value_score": round(self.value_score, 1),
            "quality_score": round(self.quality_score, 1),
            "momentum_score": round(self.momentum_score, 1),
            "volatility_score": round(self.volatility_score, 1),
            "metrics": {
                "pe_ratio": self.pe_ratio,
                "pb_ratio": self.pb_ratio,
                "roe": self.roe,
                "debt_to_equity": self.debt_to_equity,
                "profit_margin": self.profit_margin,
                "momentum_6m": self.momentum_6m,
                "beta": self.beta,
                "std_dev": self.std_dev,
            },
        }


def calculate_value_score(
    pe_ratio: float | None,
    pb_ratio: float | None,
    sector_pe: float = 15.0,
    sector_pb: float = 2.0,
) -> float:
    """Calculate value factor score (0-100).

    Lower P/E and P/B relative to sector = higher score.
    Uses percentile ranking within reasonable bounds.

    Args:
        pe_ratio: Price-to-Earnings ratio
        pb_ratio: Price-to-Book ratio
        sector_pe: Sector average P/E for comparison
        sector_pb: Sector average P/B for comparison

    Returns:
        Value score 0-100 (higher = more undervalued)
    """
    if pe_ratio is None and pb_ratio is None:
        return 50.0  # Neutral if no data

    scores = []

    if pe_ratio is not None and pe_ratio > 0:
        # Lower P/E = higher score
        # Scale: P/E of 5 = 100, P/E of 50 = 0
        pe_score = max(0, min(100, 100 - ((pe_ratio - 5) / 45) * 100))

        # Bonus if below sector average
        if pe_ratio < sector_pe:
            pe_score = min(100, pe_score + 10)
        scores.append(pe_score)

    if pb_ratio is not None and pb_ratio > 0:
        # Lower P/B = higher score
        # Scale: P/B of 0.5 = 100, P/B of 5 = 0
        pb_score = max(0, min(100, 100 - ((pb_ratio - 0.5) / 4.5) * 100))

        # Bonus if below sector average
        if pb_ratio < sector_pb:
            pb_score = min(100, pb_score + 10)
        scores.append(pb_score)

    return np.mean(scores) if scores else 50.0


def calculate_quality_score(
    roe: float | None,
    debt_to_equity: float | None,
    profit_margin: float | None,
    current_ratio: float | None = None,
) -> float:
    """Calculate quality factor score (0-100).

    High ROE, low debt, healthy margins = higher score.
    Quality is weighted 30% for beginner safety.

    Args:
        roe: Return on Equity (%)
        debt_to_equity: Debt-to-Equity ratio
        profit_margin: Net profit margin (%)
        current_ratio: Current assets / Current liabilities

    Returns:
        Quality score 0-100 (higher = better quality)
    """
    if roe is None and debt_to_equity is None and profit_margin is None:
        return 50.0  # Neutral if no data

    scores = []

    # ROE scoring (higher = better)
    # Scale: ROE of 0% = 0, ROE of 25%+ = 100
    if roe is not None:
        roe_score = max(0, min(100, (roe / 25) * 100))
        scores.append(roe_score)

    # Debt-to-Equity scoring (lower = better)
    # Scale: D/E of 0 = 100, D/E of 2+ = 0
    if debt_to_equity is not None:
        if debt_to_equity <= 0:
            de_score = 100  # No debt = perfect
        else:
            de_score = max(0, min(100, 100 - (debt_to_equity / 2) * 100))
        scores.append(de_score)

    # Profit margin scoring (higher = better)
    # Scale: Margin of 0% = 0, Margin of 20%+ = 100
    if profit_margin is not None:
        margin_score = max(0, min(100, (profit_margin / 20) * 100))
        scores.append(margin_score)

    # Current ratio scoring (1.5-2.5 is ideal)
    if current_ratio is not None:
        if 1.5 <= current_ratio <= 2.5:
            cr_score = 100
        elif current_ratio < 1.0:
            cr_score = max(0, current_ratio * 50)
        elif current_ratio > 2.5:
            cr_score = max(50, 100 - (current_ratio - 2.5) * 20)
        else:
            cr_score = 80
        scores.append(cr_score)

    return np.mean(scores) if scores else 50.0


def calculate_momentum_score(
    returns_6m: float | None,
    returns_3m: float | None = None,
    returns_1m: float | None = None,
) -> float:
    """Calculate momentum factor score (0-100).

    Positive momentum with recent acceleration = higher score.
    Uses 6-month as primary, with 3m/1m for trend confirmation.

    Args:
        returns_6m: 6-month return (%)
        returns_3m: 3-month return (%)
        returns_1m: 1-month return (%)

    Returns:
        Momentum score 0-100 (higher = stronger momentum)
    """
    if returns_6m is None:
        return 50.0  # Neutral if no data

    # Primary: 6-month momentum
    # Scale: -30% = 0, +50% = 100
    base_score = max(0, min(100, ((returns_6m + 30) / 80) * 100))

    # Trend confirmation bonus
    bonus = 0

    if returns_3m is not None:
        # Recent 3m outperforming first 3m = trend accelerating
        if returns_6m > 0 and returns_3m > (returns_6m / 2):
            bonus += 5

    if returns_1m is not None:
        # Positive recent month = momentum intact
        if returns_1m > 0:
            bonus += 5
        # Very negative recent month = momentum fading
        elif returns_1m < -10:
            bonus -= 10

    return max(0, min(100, base_score + bonus))


def calculate_volatility_score(
    beta: float | None,
    std_dev: float | None,
    max_drawdown: float | None = None,
) -> float:
    """Calculate volatility factor score (0-100).

    Low volatility = higher score (safer for beginners).
    Based on academic "low volatility anomaly" research.

    Args:
        beta: Stock beta vs market
        std_dev: Standard deviation of returns (%)
        max_drawdown: Maximum drawdown (%)

    Returns:
        Volatility score 0-100 (higher = lower volatility = safer)
    """
    if beta is None and std_dev is None:
        return 50.0  # Neutral if no data

    scores = []

    # Beta scoring (lower = better for safety)
    # Scale: Beta 0.5 = 100, Beta 2.0 = 0
    if beta is not None:
        if beta <= 0.5:
            beta_score = 100
        elif beta >= 2.0:
            beta_score = 0
        else:
            beta_score = max(0, min(100, 100 - ((beta - 0.5) / 1.5) * 100))
        scores.append(beta_score)

    # Standard deviation scoring (lower = better)
    # Scale: StdDev 10% = 100, StdDev 50% = 0
    if std_dev is not None:
        std_score = max(0, min(100, 100 - ((std_dev - 10) / 40) * 100))
        scores.append(std_score)

    # Max drawdown penalty
    if max_drawdown is not None:
        # Scale: Drawdown 10% = 100, Drawdown 50% = 0
        dd_score = max(0, min(100, 100 - ((abs(max_drawdown) - 10) / 40) * 100))
        scores.append(dd_score)

    return np.mean(scores) if scores else 50.0


def calculate_composite_score(
    value_score: float,
    quality_score: float,
    momentum_score: float,
    volatility_score: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Calculate weighted composite score (0-100).

    Default weights emphasize Quality (30%) for beginner safety.

    Args:
        value_score: Value factor score
        quality_score: Quality factor score
        momentum_score: Momentum factor score
        volatility_score: Volatility factor score
        weights: Optional custom weights

    Returns:
        Composite score 0-100
    """
    w = weights or FACTOR_WEIGHTS

    composite = (
        value_score * w.get("value", 0.25)
        + quality_score * w.get("quality", 0.30)
        + momentum_score * w.get("momentum", 0.25)
        + volatility_score * w.get("volatility", 0.20)
    )

    return round(composite, 1)


def score_stock(
    symbol: str,
    fundamentals: dict[str, Any],
    price_data: dict[str, Any],
    sector_averages: dict[str, float] | None = None,
) -> FactorScores:
    """Calculate all factor scores for a stock.

    Args:
        symbol: Stock symbol
        fundamentals: Dict with pe_ratio, pb_ratio, roe, debt_to_equity, etc.
        price_data: Dict with returns_6m, returns_3m, beta, std_dev, etc.
        sector_averages: Optional sector averages for comparison

    Returns:
        FactorScores with all individual and composite scores
    """
    sector_avg = sector_averages or {}

    # Extract metrics
    pe = fundamentals.get("pe_ratio")
    pb = fundamentals.get("pb_ratio")
    roe = fundamentals.get("roe")
    de = fundamentals.get("debt_to_equity")
    margin = fundamentals.get("profit_margin")
    current_ratio = fundamentals.get("current_ratio")

    ret_6m = price_data.get("returns_6m")
    ret_3m = price_data.get("returns_3m")
    ret_1m = price_data.get("returns_1m")
    beta = price_data.get("beta")
    std = price_data.get("std_dev")
    mdd = price_data.get("max_drawdown")

    # Calculate factor scores
    value = calculate_value_score(
        pe, pb,
        sector_avg.get("pe", 15.0),
        sector_avg.get("pb", 2.0),
    )
    quality = calculate_quality_score(roe, de, margin, current_ratio)
    momentum = calculate_momentum_score(ret_6m, ret_3m, ret_1m)
    volatility = calculate_volatility_score(beta, std, mdd)

    composite = calculate_composite_score(value, quality, momentum, volatility)

    return FactorScores(
        symbol=symbol,
        value_score=value,
        quality_score=quality,
        momentum_score=momentum,
        volatility_score=volatility,
        composite_score=composite,
        pe_ratio=pe,
        pb_ratio=pb,
        roe=roe,
        debt_to_equity=de,
        profit_margin=margin,
        momentum_6m=ret_6m,
        beta=beta,
        std_dev=std,
    )


def get_score_interpretation(score: float) -> str:
    """Get human-readable interpretation of a composite score.

    Args:
        score: Composite score 0-100

    Returns:
        Text interpretation
    """
    if score >= 80:
        return "EXCELLENT - Strong buy candidate, meets all quality criteria"
    elif score >= 70:
        return "GOOD - Solid pick, minor weaknesses but overall attractive"
    elif score >= 60:
        return "FAIR - Average, may need closer analysis before buying"
    elif score >= 50:
        return "NEUTRAL - Mixed signals, better options likely available"
    elif score >= 40:
        return "WEAK - Below average, multiple concerns present"
    else:
        return "POOR - Avoid, significant red flags in multiple factors"
