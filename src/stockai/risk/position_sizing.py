"""Position Sizing Module.

Implements the 2% risk rule used by professional traders:
- Never risk more than 2% of capital on a single trade
- Automatically calculates position size based on stop-loss distance
- Adapts for small capital (IDX lot size = 100 shares)
"""

from dataclasses import dataclass
from typing import Any

import logging

logger = logging.getLogger(__name__)

# Indonesian stock market constants
SHARES_PER_LOT = 100
MIN_LOTS = 1
DEFAULT_BROKER_FEE = 0.0015  # 0.15%
DEFAULT_TAX_RATE = 0.001  # 0.1% on sell


@dataclass
class PositionSize:
    """Calculated position size with risk metrics."""

    symbol: str
    entry_price: float
    stop_loss_price: float
    target_price: float | None

    # Position details
    lots: int
    shares: int
    position_value: float

    # Risk metrics
    risk_amount: float  # Max loss in Rupiah
    risk_percent: float  # % of capital at risk
    reward_amount: float | None  # Potential gain
    risk_reward_ratio: float | None

    # Fees
    buy_fee: float
    sell_fee: float
    total_cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
            "target_price": self.target_price,
            "lots": self.lots,
            "shares": self.shares,
            "position_value": self.position_value,
            "total_cost": self.total_cost,
            "risk_amount": self.risk_amount,
            "risk_percent": self.risk_percent,
            "reward_amount": self.reward_amount,
            "risk_reward_ratio": self.risk_reward_ratio,
            "buy_fee": self.buy_fee,
            "sell_fee": self.sell_fee,
        }


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss_price: float,
    target_price: float | None = None,
    symbol: str = "",
    max_risk_percent: float = 2.0,
    max_position_percent: float = 20.0,
    broker_fee: float = DEFAULT_BROKER_FEE,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> PositionSize:
    """Calculate optimal position size using the 2% rule.

    The 2% rule ensures that if your stop-loss is hit, you lose
    at most 2% of your total capital. This protects against
    catastrophic losses and allows for multiple losing trades
    without significant portfolio damage.

    Args:
        capital: Total capital in Rupiah
        entry_price: Planned entry price per share
        stop_loss_price: Stop-loss price per share
        target_price: Optional target price per share
        symbol: Stock symbol for reference
        max_risk_percent: Maximum risk per trade (default 2%)
        max_position_percent: Maximum position size as % of capital
        broker_fee: Broker fee rate
        tax_rate: Tax rate on sell

    Returns:
        PositionSize with calculated lots and risk metrics
    """
    # Validate inputs
    if stop_loss_price >= entry_price:
        raise ValueError("Stop-loss must be below entry price for long positions")

    if capital <= 0 or entry_price <= 0:
        raise ValueError("Capital and entry price must be positive")

    # Calculate risk per share
    risk_per_share = entry_price - stop_loss_price

    # Maximum risk amount (2% of capital by default)
    max_risk_amount = capital * (max_risk_percent / 100)

    # Calculate shares based on risk
    shares_by_risk = int(max_risk_amount / risk_per_share)

    # Round down to complete lots
    lots_by_risk = shares_by_risk // SHARES_PER_LOT

    # Also check position size limit (max 20% of capital in one stock)
    max_position_value = capital * (max_position_percent / 100)
    lots_by_position = int(max_position_value / (entry_price * SHARES_PER_LOT))

    # Take the smaller of the two limits
    lots = max(MIN_LOTS, min(lots_by_risk, lots_by_position))
    shares = lots * SHARES_PER_LOT

    # Calculate values
    position_value = shares * entry_price
    buy_fee = position_value * broker_fee
    sell_fee = position_value * (broker_fee + tax_rate)  # Estimate based on entry
    total_cost = position_value + buy_fee

    # Actual risk amount
    risk_amount = shares * risk_per_share
    risk_percent = (risk_amount / capital) * 100

    # Reward calculation
    reward_amount = None
    risk_reward_ratio = None
    if target_price is not None and target_price > entry_price:
        reward_per_share = target_price - entry_price
        reward_amount = shares * reward_per_share
        risk_reward_ratio = round(reward_amount / risk_amount, 2) if risk_amount > 0 else None

    return PositionSize(
        symbol=symbol,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        target_price=target_price,
        lots=lots,
        shares=shares,
        position_value=position_value,
        risk_amount=risk_amount,
        risk_percent=round(risk_percent, 2),
        reward_amount=reward_amount,
        risk_reward_ratio=risk_reward_ratio,
        buy_fee=round(buy_fee, 0),
        sell_fee=round(sell_fee, 0),
        total_cost=round(total_cost, 0),
    )


def calculate_max_loss(
    capital: float,
    max_risk_percent: float = 2.0,
) -> float:
    """Calculate maximum acceptable loss per trade.

    Args:
        capital: Total capital in Rupiah
        max_risk_percent: Maximum risk percentage

    Returns:
        Maximum loss amount in Rupiah
    """
    return capital * (max_risk_percent / 100)


def calculate_stop_loss_price(
    entry_price: float,
    capital: float,
    lots: int,
    max_risk_percent: float = 2.0,
) -> float:
    """Calculate stop-loss price for given position size.

    Reverse calculation: given a position size, what stop-loss
    ensures we don't risk more than 2%?

    Args:
        entry_price: Entry price per share
        capital: Total capital
        lots: Number of lots
        max_risk_percent: Maximum risk percentage

    Returns:
        Suggested stop-loss price
    """
    shares = lots * SHARES_PER_LOT
    max_risk = capital * (max_risk_percent / 100)
    max_loss_per_share = max_risk / shares
    stop_loss = entry_price - max_loss_per_share

    return round(stop_loss, 0)


def format_position_size_for_display(pos: PositionSize, capital: float) -> str:
    """Format position size for CLI display.

    Args:
        pos: PositionSize to format
        capital: Total capital for context

    Returns:
        Formatted string
    """
    lines = [
        f"📊 Position Size Calculator: {pos.symbol}",
        f"",
        f"Entry Price:     Rp {pos.entry_price:>12,.0f}",
        f"Stop-Loss:       Rp {pos.stop_loss_price:>12,.0f} ({((pos.entry_price - pos.stop_loss_price) / pos.entry_price * 100):.1f}% below)",
    ]

    if pos.target_price:
        lines.append(f"Target:          Rp {pos.target_price:>12,.0f} ({((pos.target_price - pos.entry_price) / pos.entry_price * 100):.1f}% above)")

    lines.extend([
        f"",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"Recommended Position:",
        f"  Lots:          {pos.lots:>12} lot{'s' if pos.lots > 1 else ''}",
        f"  Shares:        {pos.shares:>12,} shares",
        f"  Position Value: Rp {pos.position_value:>10,.0f}",
        f"  + Buy Fee:      Rp {pos.buy_fee:>10,.0f}",
        f"  Total Cost:     Rp {pos.total_cost:>10,.0f}",
        f"",
        f"Risk Analysis:",
        f"  Max Loss:       Rp {pos.risk_amount:>10,.0f}",
        f"  Risk %:         {pos.risk_percent:>12.1f}% of capital",
        f"  Position %:     {(pos.position_value / capital * 100):>12.1f}% of capital",
    ])

    if pos.reward_amount:
        lines.extend([
            f"  Potential Gain: Rp {pos.reward_amount:>10,.0f}",
            f"  Risk/Reward:    1:{pos.risk_reward_ratio:>10.1f}",
        ])

    # Add recommendation
    lines.append(f"")
    if pos.risk_percent <= 2.0:
        lines.append("✅ Position size is within 2% risk rule")
    else:
        lines.append("⚠️ Position size exceeds 2% risk - consider reducing")

    if pos.risk_reward_ratio and pos.risk_reward_ratio >= 2.0:
        lines.append("✅ Risk/Reward ratio is favorable (>= 1:2)")
    elif pos.risk_reward_ratio:
        lines.append("⚠️ Risk/Reward ratio is low - consider better target")

    return "\n".join(lines)


# Quick helper for small capital investors
def quick_position_size(
    capital: float,
    stock_price: float,
    stop_loss_pct: float = 8.0,
    target_pct: float = 15.0,
    symbol: str = "",
) -> PositionSize:
    """Quick position sizing with percentage-based stop/target.

    Convenience function for beginners who think in percentages.

    Args:
        capital: Total capital
        stock_price: Current stock price (entry price)
        stop_loss_pct: Stop-loss percentage below entry (default 8%)
        target_pct: Target percentage above entry (default 15%)
        symbol: Stock symbol

    Returns:
        PositionSize calculation
    """
    stop_loss_price = stock_price * (1 - stop_loss_pct / 100)
    target_price = stock_price * (1 + target_pct / 100)

    return calculate_position_size(
        capital=capital,
        entry_price=stock_price,
        stop_loss_price=stop_loss_price,
        target_price=target_price,
        symbol=symbol,
    )
