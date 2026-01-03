"""Diversification Check Module.

Ensures portfolio doesn't concentrate risk:
- Max 20% per stock
- Max 40% per sector
- Minimum 3-5 stocks for small capital
"""

from dataclasses import dataclass, field
from typing import Any

import logging

logger = logging.getLogger(__name__)


@dataclass
class DiversificationLimits:
    """Diversification rules for portfolio construction."""

    max_per_stock: float = 0.20  # Max 20% per stock
    max_per_sector: float = 0.40  # Max 40% per sector
    min_stocks: int = 3  # Minimum 3 stocks
    max_stocks: int = 10  # Maximum 10 stocks for manageability
    min_sectors: int = 2  # Minimum 2 sectors


@dataclass
class DiversificationIssue:
    """A diversification violation."""

    issue_type: str  # 'stock_concentration', 'sector_concentration', 'insufficient_diversification'
    severity: str  # 'warning', 'violation'
    message: str
    current_value: float
    limit_value: float
    affected_items: list[str] = field(default_factory=list)


@dataclass
class DiversificationCheck:
    """Result of diversification analysis."""

    is_compliant: bool
    issues: list[DiversificationIssue]
    stock_weights: dict[str, float]
    sector_weights: dict[str, float]
    total_stocks: int
    total_sectors: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_compliant": self.is_compliant,
            "issues": [
                {
                    "type": i.issue_type,
                    "severity": i.severity,
                    "message": i.message,
                    "current": i.current_value,
                    "limit": i.limit_value,
                    "affected": i.affected_items,
                }
                for i in self.issues
            ],
            "stock_weights": self.stock_weights,
            "sector_weights": self.sector_weights,
            "total_stocks": self.total_stocks,
            "total_sectors": self.total_sectors,
        }


def check_diversification(
    positions: list[dict[str, Any]],
    limits: DiversificationLimits | None = None,
) -> DiversificationCheck:
    """Check portfolio diversification against limits.

    Args:
        positions: List of positions with 'symbol', 'value', 'sector'
        limits: Diversification rules (uses defaults if not provided)

    Returns:
        DiversificationCheck with compliance status and issues
    """
    limits = limits or DiversificationLimits()
    issues = []

    if not positions:
        return DiversificationCheck(
            is_compliant=True,
            issues=[],
            stock_weights={},
            sector_weights={},
            total_stocks=0,
            total_sectors=0,
        )

    # Calculate total portfolio value
    total_value = sum(p.get("value", 0) for p in positions)

    if total_value <= 0:
        return DiversificationCheck(
            is_compliant=True,
            issues=[],
            stock_weights={},
            sector_weights={},
            total_stocks=0,
            total_sectors=0,
        )

    # Calculate stock weights
    stock_weights = {}
    for p in positions:
        symbol = p.get("symbol", "UNKNOWN")
        weight = p.get("value", 0) / total_value
        stock_weights[symbol] = round(weight * 100, 1)

    # Calculate sector weights
    sector_totals: dict[str, float] = {}
    for p in positions:
        sector = p.get("sector", "Unknown")
        sector_totals[sector] = sector_totals.get(sector, 0) + p.get("value", 0)

    sector_weights = {
        s: round(v / total_value * 100, 1)
        for s, v in sector_totals.items()
    }

    total_stocks = len(stock_weights)
    total_sectors = len(sector_weights)

    # Check stock concentration
    for symbol, weight in stock_weights.items():
        weight_decimal = weight / 100
        if weight_decimal > limits.max_per_stock:
            issues.append(
                DiversificationIssue(
                    issue_type="stock_concentration",
                    severity="violation",
                    message=f"{symbol} is {weight:.1f}% of portfolio (max {limits.max_per_stock * 100:.0f}%)",
                    current_value=weight,
                    limit_value=limits.max_per_stock * 100,
                    affected_items=[symbol],
                )
            )
        elif weight_decimal > limits.max_per_stock * 0.8:
            # Warning if approaching limit
            issues.append(
                DiversificationIssue(
                    issue_type="stock_concentration",
                    severity="warning",
                    message=f"{symbol} is {weight:.1f}% of portfolio (approaching {limits.max_per_stock * 100:.0f}% limit)",
                    current_value=weight,
                    limit_value=limits.max_per_stock * 100,
                    affected_items=[symbol],
                )
            )

    # Check sector concentration
    for sector, weight in sector_weights.items():
        weight_decimal = weight / 100
        if weight_decimal > limits.max_per_sector:
            stocks_in_sector = [
                p.get("symbol", "") for p in positions
                if p.get("sector") == sector
            ]
            issues.append(
                DiversificationIssue(
                    issue_type="sector_concentration",
                    severity="violation",
                    message=f"Sector '{sector}' is {weight:.1f}% of portfolio (max {limits.max_per_sector * 100:.0f}%)",
                    current_value=weight,
                    limit_value=limits.max_per_sector * 100,
                    affected_items=stocks_in_sector,
                )
            )

    # Check minimum diversification
    if total_stocks < limits.min_stocks:
        issues.append(
            DiversificationIssue(
                issue_type="insufficient_diversification",
                severity="warning",
                message=f"Portfolio has only {total_stocks} stocks (recommend minimum {limits.min_stocks})",
                current_value=total_stocks,
                limit_value=limits.min_stocks,
            )
        )

    if total_sectors < limits.min_sectors:
        issues.append(
            DiversificationIssue(
                issue_type="insufficient_diversification",
                severity="warning",
                message=f"Portfolio has only {total_sectors} sector(s) (recommend minimum {limits.min_sectors})",
                current_value=total_sectors,
                limit_value=limits.min_sectors,
            )
        )

    # Check if over-diversified
    if total_stocks > limits.max_stocks:
        issues.append(
            DiversificationIssue(
                issue_type="over_diversification",
                severity="warning",
                message=f"Portfolio has {total_stocks} stocks (max recommended {limits.max_stocks} for manageability)",
                current_value=total_stocks,
                limit_value=limits.max_stocks,
            )
        )

    # Determine compliance
    has_violations = any(i.severity == "violation" for i in issues)

    return DiversificationCheck(
        is_compliant=not has_violations,
        issues=issues,
        stock_weights=stock_weights,
        sector_weights=sector_weights,
        total_stocks=total_stocks,
        total_sectors=total_sectors,
    )


def suggest_rebalance(
    positions: list[dict[str, Any]],
    target_weight: float = 0.20,
) -> list[dict[str, Any]]:
    """Suggest trades to rebalance portfolio.

    Args:
        positions: Current positions
        target_weight: Target weight per stock

    Returns:
        List of suggested trades
    """
    if not positions:
        return []

    total_value = sum(p.get("value", 0) for p in positions)
    suggestions = []

    for p in positions:
        symbol = p.get("symbol", "")
        current_value = p.get("value", 0)
        current_weight = current_value / total_value if total_value > 0 else 0

        if current_weight > target_weight:
            # Overweight - suggest trim
            target_value = total_value * target_weight
            excess = current_value - target_value
            suggestions.append({
                "symbol": symbol,
                "action": "TRIM",
                "current_weight": round(current_weight * 100, 1),
                "target_weight": round(target_weight * 100, 1),
                "excess_value": round(excess, 0),
                "message": f"Consider selling Rp {excess:,.0f} of {symbol} to reduce from {current_weight * 100:.1f}% to {target_weight * 100:.0f}%",
            })

    return suggestions


def format_diversification_for_display(check: DiversificationCheck) -> str:
    """Format diversification check for CLI display.

    Args:
        check: DiversificationCheck result

    Returns:
        Formatted string
    """
    lines = []

    # Header
    if check.is_compliant:
        lines.append("✅ Portfolio Diversification: COMPLIANT")
    else:
        lines.append("⚠️ Portfolio Diversification: ISSUES FOUND")

    lines.append("")

    # Stock weights
    lines.append(f"📊 Stock Allocation ({check.total_stocks} stocks):")
    for symbol, weight in sorted(check.stock_weights.items(), key=lambda x: -x[1]):
        bar_len = int(weight / 5)
        bar = "█" * bar_len
        lines.append(f"   {symbol:8} {weight:5.1f}% {bar}")

    lines.append("")

    # Sector weights
    lines.append(f"🏭 Sector Allocation ({check.total_sectors} sectors):")
    for sector, weight in sorted(check.sector_weights.items(), key=lambda x: -x[1]):
        bar_len = int(weight / 5)
        bar = "█" * bar_len
        lines.append(f"   {sector:15} {weight:5.1f}% {bar}")

    # Issues
    if check.issues:
        lines.append("")
        lines.append("⚠️ Issues:")
        for issue in check.issues:
            icon = "🔴" if issue.severity == "violation" else "🟡"
            lines.append(f"   {icon} {issue.message}")

    return "\n".join(lines)
