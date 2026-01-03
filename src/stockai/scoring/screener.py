"""Stock Screening Module.

Implements systematic filtering for Indonesian stocks:
1. Universe definition (IDX30/LQ45)
2. Liquidity filters
3. Fundamental filters
4. Technical filters
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class Universe(Enum):
    """Stock universe options."""

    IDX30 = "IDX30"  # Top 30 most liquid
    LQ45 = "LQ45"  # Top 45 most liquid
    IDXHIDIV20 = "IDXHIDIV20"  # High dividend
    ALL = "ALL"  # All stocks (not recommended for beginners)


@dataclass
class ScreeningCriteria:
    """Criteria for filtering stocks.

    Defaults are set for beginner safety and small capital.
    """

    # Universe
    universe: Universe = Universe.IDX30

    # Liquidity filters
    min_avg_volume: int = 1_000_000  # Min 1M shares daily
    min_market_cap: float = 1_000_000_000_000  # Min 1T Rupiah

    # Fundamental filters (Quality focus)
    min_roe: float = 10.0  # Minimum 10% ROE
    max_debt_to_equity: float = 1.5  # Max 1.5x debt ratio
    min_profit_margin: float = 5.0  # Minimum 5% net margin
    max_pe_ratio: float = 30.0  # Max P/E of 30
    min_pe_ratio: float = 3.0  # Min P/E of 3 (avoid value traps)

    # Technical filters
    max_volatility: float = 40.0  # Max 40% annual volatility
    max_beta: float = 1.5  # Max beta of 1.5
    min_momentum_6m: float = -20.0  # Not more than 20% down in 6M

    # Score filters
    min_composite_score: float = 50.0  # Minimum passing score

    def to_dict(self) -> dict[str, Any]:
        return {
            "universe": self.universe.value,
            "min_avg_volume": self.min_avg_volume,
            "min_market_cap": self.min_market_cap,
            "min_roe": self.min_roe,
            "max_debt_to_equity": self.max_debt_to_equity,
            "min_profit_margin": self.min_profit_margin,
            "max_pe_ratio": self.max_pe_ratio,
            "min_pe_ratio": self.min_pe_ratio,
            "max_volatility": self.max_volatility,
            "max_beta": self.max_beta,
            "min_momentum_6m": self.min_momentum_6m,
            "min_composite_score": self.min_composite_score,
        }


@dataclass
class ScreeningResult:
    """Result of stock screening."""

    symbol: str
    passed: bool
    failed_criteria: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


class StockScreener:
    """Screen stocks based on systematic criteria."""

    def __init__(self, criteria: ScreeningCriteria | None = None):
        """Initialize screener.

        Args:
            criteria: Screening criteria (uses defaults if not provided)
        """
        self.criteria = criteria or ScreeningCriteria()

    def screen_stock(
        self,
        symbol: str,
        fundamentals: dict[str, Any],
        technicals: dict[str, Any],
        score: float | None = None,
    ) -> ScreeningResult:
        """Screen a single stock against criteria.

        Args:
            symbol: Stock symbol
            fundamentals: Fundamental metrics
            technicals: Technical metrics
            score: Optional composite score

        Returns:
            ScreeningResult with pass/fail and reasons
        """
        failed = []
        c = self.criteria

        # Liquidity checks
        vol = technicals.get("avg_volume", 0)
        if vol < c.min_avg_volume:
            failed.append(f"Volume {vol:,.0f} < {c.min_avg_volume:,.0f}")

        mcap = fundamentals.get("market_cap", 0)
        if mcap < c.min_market_cap:
            failed.append(f"Market cap Rp {mcap/1e12:.1f}T < Rp {c.min_market_cap/1e12:.0f}T")

        # Fundamental checks
        roe = fundamentals.get("roe")
        if roe is not None and roe < c.min_roe:
            failed.append(f"ROE {roe:.1f}% < {c.min_roe:.0f}%")

        de = fundamentals.get("debt_to_equity")
        if de is not None and de > c.max_debt_to_equity:
            failed.append(f"D/E {de:.2f} > {c.max_debt_to_equity:.1f}")

        margin = fundamentals.get("profit_margin")
        if margin is not None and margin < c.min_profit_margin:
            failed.append(f"Margin {margin:.1f}% < {c.min_profit_margin:.0f}%")

        pe = fundamentals.get("pe_ratio")
        if pe is not None:
            if pe > c.max_pe_ratio:
                failed.append(f"P/E {pe:.1f} > {c.max_pe_ratio:.0f}")
            elif pe < c.min_pe_ratio:
                failed.append(f"P/E {pe:.1f} < {c.min_pe_ratio:.0f} (value trap risk)")

        # Technical checks
        vol_pct = technicals.get("volatility")
        if vol_pct is not None and vol_pct > c.max_volatility:
            failed.append(f"Volatility {vol_pct:.1f}% > {c.max_volatility:.0f}%")

        beta = technicals.get("beta")
        if beta is not None and beta > c.max_beta:
            failed.append(f"Beta {beta:.2f} > {c.max_beta:.1f}")

        momentum = technicals.get("returns_6m")
        if momentum is not None and momentum < c.min_momentum_6m:
            failed.append(f"6M return {momentum:.1f}% < {c.min_momentum_6m:.0f}%")

        # Score check
        if score is not None and score < c.min_composite_score:
            failed.append(f"Score {score:.1f} < {c.min_composite_score:.0f}")

        return ScreeningResult(
            symbol=symbol,
            passed=len(failed) == 0,
            failed_criteria=failed,
            data={**fundamentals, **technicals, "composite_score": score},
        )

    def get_passing_stocks(
        self,
        stocks: list[dict[str, Any]],
    ) -> list[ScreeningResult]:
        """Filter list of stocks through screening criteria.

        Args:
            stocks: List of dicts with symbol, fundamentals, technicals

        Returns:
            List of passing ScreeningResults
        """
        results = []

        for stock in stocks:
            result = self.screen_stock(
                symbol=stock.get("symbol", ""),
                fundamentals=stock.get("fundamentals", {}),
                technicals=stock.get("technicals", {}),
                score=stock.get("score"),
            )
            if result.passed:
                results.append(result)

        return results


# Preset screening criteria for different strategies
SCREENING_PRESETS = {
    "conservative": ScreeningCriteria(
        universe=Universe.IDX30,
        min_roe=15.0,
        max_debt_to_equity=1.0,
        min_profit_margin=10.0,
        max_volatility=30.0,
        max_beta=1.0,
        min_composite_score=70.0,
    ),
    "balanced": ScreeningCriteria(
        universe=Universe.IDX30,
        min_roe=10.0,
        max_debt_to_equity=1.5,
        min_profit_margin=5.0,
        max_volatility=40.0,
        max_beta=1.5,
        min_composite_score=60.0,
    ),
    "aggressive": ScreeningCriteria(
        universe=Universe.LQ45,
        min_roe=5.0,
        max_debt_to_equity=2.0,
        min_profit_margin=0.0,
        max_volatility=50.0,
        max_beta=2.0,
        min_composite_score=50.0,
    ),
}


def get_preset_criteria(preset: str) -> ScreeningCriteria:
    """Get preset screening criteria.

    Args:
        preset: One of 'conservative', 'balanced', 'aggressive'

    Returns:
        ScreeningCriteria for the preset
    """
    return SCREENING_PRESETS.get(preset.lower(), SCREENING_PRESETS["balanced"])
