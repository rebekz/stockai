"""Trade Plan Generation Module.

Generates complete trade plans with entry ranges, stop loss, and multiple
take profit levels based on support/resistance analysis.
"""

from dataclasses import dataclass


@dataclass
class TradePlanConfig:
    """Configuration for trade plan generation."""

    stop_loss_pct_below_support: float = 0.03  # 3% below support
    tp1_pct: float = 0.05  # 5% profit target
    tp2_pct: float = 0.10  # 10% profit target
    tp3_pct: float = 0.15  # 15% profit target
    tp1_sell_pct: float = 0.25  # Sell 25% at TP1
    tp2_sell_pct: float = 0.50  # Sell 50% at TP2
    tp3_sell_pct: float = 0.25  # Sell remaining 25% at TP3


@dataclass
class TradePlan:
    """Complete trade plan with entry, SL, and TPs."""

    entry_low: float
    entry_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward_ratio: float
    risk_pct: float
    summary: str


def generate_trade_plan(
    current_price: float,
    support: float | None,
    resistances: list[float] | None = None,
    config: TradePlanConfig | None = None,
) -> TradePlan:
    """Generate a complete trade plan based on support/resistance.

    Args:
        current_price: Current stock price
        support: Nearest support level (or None)
        resistances: List of resistance levels (optional)
        config: Trade plan configuration

    Returns:
        TradePlan with entry, SL, TP levels
    """
    if config is None:
        config = TradePlanConfig()

    resistances = resistances or []

    # Calculate entry range
    if support is not None:
        # Entry from 1% above support to current price
        entry_low = support * 1.01
        entry_high = current_price
    else:
        # No support found - use 2% below current price as entry low
        entry_low = current_price * 0.98
        entry_high = current_price

    # Ensure entry_low doesn't exceed entry_high
    if entry_low > entry_high:
        entry_low = entry_high * 0.98

    # Calculate stop loss
    if support is not None:
        # 3% below support
        stop_loss = support * (1 - config.stop_loss_pct_below_support)
    else:
        # 8% below current price if no support
        stop_loss = current_price * 0.92

    # Calculate take profit levels
    # Priority: use resistance levels if available, otherwise use default percentages
    mid_entry = (entry_low + entry_high) / 2

    if len(resistances) >= 3:
        # Use actual resistance levels
        take_profit_1 = resistances[0]
        take_profit_2 = resistances[1]
        take_profit_3 = resistances[2]
    elif len(resistances) == 2:
        take_profit_1 = resistances[0]
        take_profit_2 = resistances[1]
        take_profit_3 = mid_entry * (1 + config.tp3_pct)
    elif len(resistances) == 1:
        take_profit_1 = resistances[0]
        take_profit_2 = mid_entry * (1 + config.tp2_pct)
        take_profit_3 = mid_entry * (1 + config.tp3_pct)
    else:
        # Use default percentage-based TPs
        take_profit_1 = mid_entry * (1 + config.tp1_pct)
        take_profit_2 = mid_entry * (1 + config.tp2_pct)
        take_profit_3 = mid_entry * (1 + config.tp3_pct)

    # Ensure TPs are above entry and ascending
    take_profit_1 = max(take_profit_1, entry_high * 1.02)
    take_profit_2 = max(take_profit_2, take_profit_1 * 1.02)
    take_profit_3 = max(take_profit_3, take_profit_2 * 1.02)

    # Calculate risk/reward ratio
    # Risk = mid_entry - stop_loss
    # Reward = weighted average of TPs - mid_entry
    risk = mid_entry - stop_loss
    weighted_reward = (
        (take_profit_1 - mid_entry) * config.tp1_sell_pct
        + (take_profit_2 - mid_entry) * config.tp2_sell_pct
        + (take_profit_3 - mid_entry) * config.tp3_sell_pct
    )
    risk_reward_ratio = weighted_reward / risk if risk > 0 else 0

    # Calculate risk percentage from entry
    risk_pct = (risk / mid_entry) * 100 if mid_entry > 0 else 0

    # Generate summary
    summary = (
        f"Entry: {entry_low:,.0f} - {entry_high:,.0f} | "
        f"SL: {stop_loss:,.0f} ({risk_pct:.1f}% risk) | "
        f"TP1: {take_profit_1:,.0f} | TP2: {take_profit_2:,.0f} | TP3: {take_profit_3:,.0f} | "
        f"R/R: {risk_reward_ratio:.2f}"
    )

    return TradePlan(
        entry_low=float(entry_low),
        entry_high=float(entry_high),
        stop_loss=float(stop_loss),
        take_profit_1=float(take_profit_1),
        take_profit_2=float(take_profit_2),
        take_profit_3=float(take_profit_3),
        risk_reward_ratio=float(risk_reward_ratio),
        risk_pct=float(risk_pct),
        summary=summary,
    )


def calculate_position_with_plan(
    capital: float,
    trade_plan: TradePlan,
    risk_pct: float = 0.02,
) -> dict:
    """Calculate position size using 2% risk rule with trade plan.

    Args:
        capital: Total trading capital
        trade_plan: Trade plan with entry and stop loss
        risk_pct: Maximum risk per trade (default 2%)

    Returns:
        Dictionary with position sizing details
    """
    # Maximum loss allowed
    max_loss = capital * risk_pct

    # Entry price (use mid-point of entry range)
    entry_price = (trade_plan.entry_low + trade_plan.entry_high) / 2

    # Risk per share
    risk_per_share = entry_price - trade_plan.stop_loss

    if risk_per_share <= 0:
        return {
            "shares": 0,
            "lots": 0,
            "position_value": 0,
            "max_loss": 0,
            "error": "Invalid stop loss - must be below entry",
        }

    # Number of shares
    shares = int(max_loss / risk_per_share)

    # Round to lots (100 shares per lot in Indonesia)
    lots = shares // 100
    shares = lots * 100

    # Position value
    position_value = shares * entry_price

    # Actual max loss
    actual_max_loss = shares * risk_per_share

    # Position as percentage of capital
    position_pct = (position_value / capital) * 100 if capital > 0 else 0

    return {
        "shares": shares,
        "lots": lots,
        "position_value": position_value,
        "position_pct": position_pct,
        "entry_price": entry_price,
        "stop_loss": trade_plan.stop_loss,
        "max_loss": actual_max_loss,
        "risk_per_share": risk_per_share,
    }
