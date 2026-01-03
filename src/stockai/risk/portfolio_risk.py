"""Portfolio Risk Module.

Calculate portfolio-level risk metrics:
- Portfolio volatility
- Value at Risk (VaR)
- Maximum drawdown analysis
- Correlation matrix
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""

    total_value: float
    portfolio_volatility: float  # Annualized
    value_at_risk_95: float  # 95% VaR in Rupiah
    value_at_risk_99: float  # 99% VaR in Rupiah
    max_drawdown: float  # Historical max drawdown %
    sharpe_ratio: float | None  # Risk-adjusted return
    beta: float  # Portfolio beta vs market
    diversification_score: float  # 0-100

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_value": self.total_value,
            "portfolio_volatility": round(self.portfolio_volatility, 2),
            "value_at_risk_95": round(self.value_at_risk_95, 0),
            "value_at_risk_99": round(self.value_at_risk_99, 0),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2) if self.sharpe_ratio else None,
            "beta": round(self.beta, 2),
            "diversification_score": round(self.diversification_score, 1),
        }


def calculate_portfolio_risk(
    positions: list[dict[str, Any]],
    returns_data: dict[str, list[float]] | None = None,
    market_returns: list[float] | None = None,
    risk_free_rate: float = 0.06,  # ~6% for Indonesia
) -> PortfolioRisk:
    """Calculate comprehensive portfolio risk metrics.

    Args:
        positions: List of positions with 'symbol', 'value', 'weight'
        returns_data: Optional dict of symbol -> daily returns list
        market_returns: Optional market (IHSG) daily returns
        risk_free_rate: Annual risk-free rate (BI rate)

    Returns:
        PortfolioRisk with all metrics
    """
    if not positions:
        return PortfolioRisk(
            total_value=0,
            portfolio_volatility=0,
            value_at_risk_95=0,
            value_at_risk_99=0,
            max_drawdown=0,
            sharpe_ratio=None,
            beta=1.0,
            diversification_score=0,
        )

    total_value = sum(p.get("value", 0) for p in positions)

    # Calculate weights
    weights = []
    for p in positions:
        w = p.get("value", 0) / total_value if total_value > 0 else 0
        weights.append(w)

    weights = np.array(weights)

    # If we have returns data, calculate actual metrics
    if returns_data and len(returns_data) > 0:
        # Build returns matrix
        symbols = [p.get("symbol", "") for p in positions]
        returns_matrix = []

        for symbol in symbols:
            if symbol in returns_data:
                returns_matrix.append(returns_data[symbol])
            else:
                # Use zeros for missing data
                returns_matrix.append([0] * 60)

        returns_matrix = np.array(returns_matrix)

        if returns_matrix.size > 0:
            # Portfolio returns (weighted average)
            portfolio_returns = np.dot(weights, returns_matrix)

            # Volatility (annualized)
            daily_vol = np.std(portfolio_returns)
            portfolio_volatility = daily_vol * np.sqrt(252) * 100

            # Value at Risk
            var_95 = np.percentile(portfolio_returns, 5) * total_value
            var_99 = np.percentile(portfolio_returns, 1) * total_value

            # Max drawdown
            cumulative = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns)) * 100

            # Sharpe ratio
            annual_return = np.mean(portfolio_returns) * 252
            sharpe = (annual_return - risk_free_rate) / (daily_vol * np.sqrt(252)) if daily_vol > 0 else None

            # Beta (if market returns available)
            if market_returns and len(market_returns) == len(portfolio_returns):
                covariance = np.cov(portfolio_returns, market_returns)[0][1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance if market_variance > 0 else 1.0
            else:
                beta = 1.0
        else:
            # Default values when no return data
            portfolio_volatility = 25.0
            var_95 = -total_value * 0.03
            var_99 = -total_value * 0.05
            max_drawdown = 15.0
            sharpe = None
            beta = 1.0
    else:
        # Estimate without returns data
        portfolio_volatility = 25.0  # Typical IDX volatility
        var_95 = -total_value * 0.03  # ~3% daily loss at 95%
        var_99 = -total_value * 0.05  # ~5% daily loss at 99%
        max_drawdown = 15.0
        sharpe = None
        beta = 1.0

    # Diversification score based on concentration
    # HHI (Herfindahl-Hirschman Index) based
    hhi = sum(w ** 2 for w in weights)
    # Convert: HHI of 1 (one stock) = 0, HHI of 1/n (equal weight) = 100
    n = len(positions)
    if n > 1:
        min_hhi = 1 / n
        diversification_score = (1 - hhi) / (1 - min_hhi) * 100 if hhi < 1 else 0
    else:
        diversification_score = 0

    return PortfolioRisk(
        total_value=total_value,
        portfolio_volatility=portfolio_volatility,
        value_at_risk_95=abs(var_95),
        value_at_risk_99=abs(var_99),
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        beta=beta,
        diversification_score=diversification_score,
    )


def format_portfolio_risk_for_display(risk: PortfolioRisk) -> str:
    """Format portfolio risk for CLI display.

    Args:
        risk: PortfolioRisk to format

    Returns:
        Formatted string
    """
    lines = [
        "📈 Portfolio Risk Analysis",
        "",
        f"Total Value:         Rp {risk.total_value:>15,.0f}",
        "",
        "Risk Metrics:",
        f"  Volatility:        {risk.portfolio_volatility:>15.1f}% (annualized)",
        f"  Beta:              {risk.beta:>15.2f} (vs IHSG)",
        f"  Max Drawdown:      {risk.max_drawdown:>15.1f}%",
        "",
        "Value at Risk (daily):",
        f"  95% VaR:           Rp {risk.value_at_risk_95:>12,.0f}",
        f"  99% VaR:           Rp {risk.value_at_risk_99:>12,.0f}",
    ]

    if risk.sharpe_ratio:
        lines.append("")
        lines.append(f"Sharpe Ratio:        {risk.sharpe_ratio:>15.2f}")

    lines.extend([
        "",
        f"Diversification:     {risk.diversification_score:>15.0f}/100",
    ])

    # Risk level interpretation
    lines.append("")
    if risk.portfolio_volatility < 20:
        lines.append("📊 Risk Level: LOW - Conservative portfolio")
    elif risk.portfolio_volatility < 35:
        lines.append("📊 Risk Level: MODERATE - Balanced portfolio")
    else:
        lines.append("📊 Risk Level: HIGH - Aggressive portfolio")

    return "\n".join(lines)
