"""Daily Briefing Module.

Morning and Evening briefings for 15-minute daily workflow:
- Morning (before market open): Alerts, signals, filtered news
- Evening (after market close): P&L summary, score changes, alerts review
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import pytz

import logging

logger = logging.getLogger(__name__)

TIMEZONE = pytz.timezone("Asia/Jakarta")


@dataclass
class Alert:
    """An actionable alert for the user."""

    alert_type: str  # 'stop_loss', 'target', 'signal', 'score_change', 'news'
    severity: str  # 'critical', 'warning', 'info'
    symbol: str
    message: str
    action_required: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(TIMEZONE))


@dataclass
class MorningBriefing:
    """Morning briefing before market open (~08:30 WIB)."""

    date: datetime
    market_status: str  # 'pre_open', 'open', 'closed'

    # Critical alerts (stop-losses, triggered targets)
    critical_alerts: list[Alert] = field(default_factory=list)

    # New signals from overnight analysis
    new_signals: list[dict[str, Any]] = field(default_factory=list)

    # Portfolio snapshot
    portfolio_value: float = 0
    portfolio_pnl: float = 0
    portfolio_pnl_pct: float = 0

    # Positions requiring attention
    positions_near_stop: list[dict[str, Any]] = field(default_factory=list)
    positions_near_target: list[dict[str, Any]] = field(default_factory=list)

    # Filtered news (only material news, not noise)
    relevant_news: list[dict[str, Any]] = field(default_factory=list)

    # Today's watchlist
    watchlist: list[str] = field(default_factory=list)

    # Estimated reading time
    reading_time_minutes: int = 5


@dataclass
class EveningBriefing:
    """Evening briefing after market close (~16:30 WIB)."""

    date: datetime

    # Today's performance
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float

    # Trade activity
    trades_executed: list[dict[str, Any]] = field(default_factory=list)
    trades_count: int = 0

    # Score changes from morning
    score_changes: list[dict[str, Any]] = field(default_factory=list)

    # Positions summary
    positions_count: int = 0
    top_gainers: list[dict[str, Any]] = field(default_factory=list)
    top_losers: list[dict[str, Any]] = field(default_factory=list)

    # Market summary
    ihsg_change: float = 0
    ihsg_change_pct: float = 0
    sector_performance: dict[str, float] = field(default_factory=dict)

    # Tomorrow's focus
    alerts_for_tomorrow: list[str] = field(default_factory=list)

    # Estimated reading time
    reading_time_minutes: int = 5


def generate_morning_briefing(
    portfolio: dict[str, Any] | None = None,
    watchlist: list[str] | None = None,
    previous_scores: dict[str, float] | None = None,
) -> MorningBriefing:
    """Generate morning briefing for the user.

    This is the first thing a passive investor checks each morning.
    Should be scannable in 5-7 minutes.

    Args:
        portfolio: Current portfolio data with positions
        watchlist: Stocks to monitor
        previous_scores: Previous day's composite scores for comparison

    Returns:
        MorningBriefing with all relevant information
    """
    now = datetime.now(TIMEZONE)
    hour = now.hour

    # Determine market status
    if hour < 9:
        market_status = "pre_open"
    elif hour < 16:
        market_status = "open"
    else:
        market_status = "closed"

    briefing = MorningBriefing(
        date=now,
        market_status=market_status,
    )

    # Process portfolio if provided
    if portfolio:
        positions = portfolio.get("positions", {})
        cash = portfolio.get("cash", 0)

        # Calculate portfolio value
        positions_value = sum(
            p.get("current_value", p.get("shares", 0) * p.get("avg_price", 0))
            for p in positions.values()
        )
        briefing.portfolio_value = positions_value + cash

        # Calculate P&L
        initial = portfolio.get("initial_capital", briefing.portfolio_value)
        briefing.portfolio_pnl = briefing.portfolio_value - initial
        briefing.portfolio_pnl_pct = (briefing.portfolio_pnl / initial * 100) if initial > 0 else 0

        # Check positions near stop-loss (within 3% of stop)
        for symbol, pos in positions.items():
            current = pos.get("current_price", pos.get("avg_price", 0))
            stop_loss = pos.get("stop_loss")
            target = pos.get("target")

            if stop_loss and current > 0:
                distance_to_stop = (current - stop_loss) / current * 100
                if distance_to_stop < 3:
                    briefing.positions_near_stop.append({
                        "symbol": symbol,
                        "current_price": current,
                        "stop_loss": stop_loss,
                        "distance_pct": round(distance_to_stop, 1),
                    })

                    if distance_to_stop <= 0:
                        briefing.critical_alerts.append(Alert(
                            alert_type="stop_loss",
                            severity="critical",
                            symbol=symbol,
                            message=f"STOP-LOSS HIT: {symbol} at Rp {current:,.0f} (stop: Rp {stop_loss:,.0f})",
                            action_required="Consider selling to limit loss",
                        ))

            if target and current > 0:
                distance_to_target = (target - current) / current * 100
                if distance_to_target < 3:
                    briefing.positions_near_target.append({
                        "symbol": symbol,
                        "current_price": current,
                        "target": target,
                        "distance_pct": round(distance_to_target, 1),
                    })

                    if distance_to_target <= 0:
                        briefing.critical_alerts.append(Alert(
                            alert_type="target",
                            severity="info",
                            symbol=symbol,
                            message=f"TARGET REACHED: {symbol} at Rp {current:,.0f} (target: Rp {target:,.0f})",
                            action_required="Consider taking profits",
                        ))

    # Set watchlist
    if watchlist:
        briefing.watchlist = watchlist
    else:
        # Default IDX30 blue chips
        briefing.watchlist = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII"]

    # Estimate reading time
    alert_count = len(briefing.critical_alerts)
    positions_count = len(briefing.positions_near_stop) + len(briefing.positions_near_target)
    news_count = len(briefing.relevant_news)
    briefing.reading_time_minutes = max(3, min(10, 3 + alert_count + positions_count // 2 + news_count // 3))

    return briefing


def generate_evening_briefing(
    portfolio: dict[str, Any] | None = None,
    trades_today: list[dict[str, Any]] | None = None,
    morning_scores: dict[str, float] | None = None,
    current_scores: dict[str, float] | None = None,
) -> EveningBriefing:
    """Generate evening briefing after market close.

    This is the end-of-day summary for the passive investor.
    Should be scannable in 5-7 minutes.

    Args:
        portfolio: Current portfolio state
        trades_today: Trades executed today
        morning_scores: Composite scores from morning
        current_scores: Current composite scores

    Returns:
        EveningBriefing with day's summary
    """
    now = datetime.now(TIMEZONE)

    briefing = EveningBriefing(
        date=now,
        portfolio_value=0,
        daily_pnl=0,
        daily_pnl_pct=0,
    )

    # Process portfolio
    if portfolio:
        positions = portfolio.get("positions", {})
        cash = portfolio.get("cash", 0)

        positions_value = sum(
            p.get("current_value", p.get("shares", 0) * p.get("avg_price", 0))
            for p in positions.values()
        )
        briefing.portfolio_value = positions_value + cash
        briefing.positions_count = len(positions)

        # Daily P&L (simplified - would need yesterday's value in production)
        initial = portfolio.get("initial_capital", briefing.portfolio_value)
        briefing.daily_pnl = briefing.portfolio_value - initial
        briefing.daily_pnl_pct = (briefing.daily_pnl / initial * 100) if initial > 0 else 0

        # Top gainers and losers
        position_performance = []
        for symbol, pos in positions.items():
            current = pos.get("current_price", 0)
            avg = pos.get("avg_price", 0)
            if avg > 0:
                pnl_pct = (current - avg) / avg * 100
                position_performance.append({
                    "symbol": symbol,
                    "current_price": current,
                    "avg_price": avg,
                    "pnl_pct": round(pnl_pct, 2),
                })

        position_performance.sort(key=lambda x: x["pnl_pct"], reverse=True)
        briefing.top_gainers = position_performance[:3]
        briefing.top_losers = position_performance[-3:][::-1] if len(position_performance) >= 3 else []

    # Process trades
    if trades_today:
        briefing.trades_executed = trades_today
        briefing.trades_count = len(trades_today)

    # Score changes
    if morning_scores and current_scores:
        for symbol, current_score in current_scores.items():
            morning_score = morning_scores.get(symbol, current_score)
            change = current_score - morning_score
            if abs(change) >= 5:  # Only significant changes
                briefing.score_changes.append({
                    "symbol": symbol,
                    "morning_score": round(morning_score, 1),
                    "evening_score": round(current_score, 1),
                    "change": round(change, 1),
                    "direction": "up" if change > 0 else "down",
                })

    # Generate alerts for tomorrow
    if briefing.score_changes:
        improving = [s for s in briefing.score_changes if s["direction"] == "up"]
        declining = [s for s in briefing.score_changes if s["direction"] == "down"]

        if improving:
            briefing.alerts_for_tomorrow.append(
                f"Watch {', '.join(s['symbol'] for s in improving[:3])}: scores improving"
            )
        if declining:
            briefing.alerts_for_tomorrow.append(
                f"Monitor {', '.join(s['symbol'] for s in declining[:3])}: scores declining"
            )

    # Estimate reading time
    briefing.reading_time_minutes = max(3, min(10, 5 + briefing.trades_count + len(briefing.score_changes) // 2))

    return briefing


def format_morning_briefing(briefing: MorningBriefing) -> str:
    """Format morning briefing for display.

    Args:
        briefing: MorningBriefing to format

    Returns:
        Formatted string for CLI
    """
    lines = [
        "═" * 60,
        f"☀️  MORNING BRIEFING - {briefing.date.strftime('%A, %d %B %Y')}",
        f"    Market: {briefing.market_status.upper()} | Est. read: {briefing.reading_time_minutes} min",
        "═" * 60,
    ]

    # Critical alerts first
    if briefing.critical_alerts:
        lines.extend([
            "",
            "🚨 CRITICAL ALERTS",
            "─" * 40,
        ])
        for alert in briefing.critical_alerts:
            icon = "🔴" if alert.severity == "critical" else "🟡"
            lines.append(f"  {icon} {alert.message}")
            if alert.action_required:
                lines.append(f"     → Action: {alert.action_required}")
    else:
        lines.extend([
            "",
            "✅ No critical alerts this morning",
        ])

    # Portfolio snapshot
    lines.extend([
        "",
        "📊 PORTFOLIO SNAPSHOT",
        "─" * 40,
        f"  Total Value:  Rp {briefing.portfolio_value:>15,.0f}",
        f"  Total P&L:    Rp {briefing.portfolio_pnl:>15,.0f} ({briefing.portfolio_pnl_pct:+.2f}%)",
    ])

    # Positions near stop/target
    if briefing.positions_near_stop:
        lines.extend([
            "",
            "⚠️ NEAR STOP-LOSS (within 3%)",
        ])
        for pos in briefing.positions_near_stop:
            lines.append(f"  • {pos['symbol']}: {pos['distance_pct']:.1f}% above stop")

    if briefing.positions_near_target:
        lines.extend([
            "",
            "🎯 NEAR TARGET (within 3%)",
        ])
        for pos in briefing.positions_near_target:
            lines.append(f"  • {pos['symbol']}: {pos['distance_pct']:.1f}% below target")

    # New signals
    if briefing.new_signals:
        lines.extend([
            "",
            "📢 NEW SIGNALS",
            "─" * 40,
        ])
        for signal in briefing.new_signals[:5]:
            sig_type = signal.get("signal_type", "UNKNOWN")
            symbol = signal.get("symbol", "?")
            confidence = signal.get("confidence", 0)
            lines.append(f"  {sig_type:10} {symbol} (confidence: {confidence:.0f}%)")

    # Today's watchlist
    if briefing.watchlist:
        lines.extend([
            "",
            f"👀 TODAY'S WATCHLIST: {', '.join(briefing.watchlist[:5])}",
        ])

    lines.append("")
    lines.append("═" * 60)

    return "\n".join(lines)


def format_evening_briefing(briefing: EveningBriefing) -> str:
    """Format evening briefing for display.

    Args:
        briefing: EveningBriefing to format

    Returns:
        Formatted string for CLI
    """
    lines = [
        "═" * 60,
        f"🌙 EVENING BRIEFING - {briefing.date.strftime('%A, %d %B %Y')}",
        f"   Market Closed | Est. read: {briefing.reading_time_minutes} min",
        "═" * 60,
    ]

    # Daily performance
    pnl_icon = "📈" if briefing.daily_pnl >= 0 else "📉"
    lines.extend([
        "",
        f"{pnl_icon} TODAY'S PERFORMANCE",
        "─" * 40,
        f"  Portfolio Value:  Rp {briefing.portfolio_value:>15,.0f}",
        f"  Daily P&L:        Rp {briefing.daily_pnl:>15,.0f} ({briefing.daily_pnl_pct:+.2f}%)",
        f"  Positions:        {briefing.positions_count:>15} stocks",
    ])

    # Top gainers/losers
    if briefing.top_gainers:
        lines.extend([
            "",
            "🏆 TOP GAINERS",
        ])
        for pos in briefing.top_gainers[:3]:
            if pos["pnl_pct"] > 0:
                lines.append(f"  📈 {pos['symbol']}: +{pos['pnl_pct']:.2f}%")

    if briefing.top_losers:
        lines.extend([
            "",
            "📉 TOP LOSERS",
        ])
        for pos in briefing.top_losers[:3]:
            if pos["pnl_pct"] < 0:
                lines.append(f"  📉 {pos['symbol']}: {pos['pnl_pct']:.2f}%")

    # Trades executed
    if briefing.trades_executed:
        lines.extend([
            "",
            f"📋 TRADES EXECUTED: {briefing.trades_count}",
            "─" * 40,
        ])
        for trade in briefing.trades_executed[:5]:
            action = trade.get("action", "?")
            symbol = trade.get("symbol", "?")
            lots = trade.get("lots", 0)
            price = trade.get("price", 0)
            icon = "🟢" if action == "BUY" else "🔴"
            lines.append(f"  {icon} {action} {lots} lot {symbol} @ Rp {price:,.0f}")
    else:
        lines.extend([
            "",
            "📋 No trades executed today",
        ])

    # Score changes
    if briefing.score_changes:
        lines.extend([
            "",
            "📊 SCORE CHANGES (>5 points)",
            "─" * 40,
        ])
        for change in briefing.score_changes[:5]:
            icon = "⬆️" if change["direction"] == "up" else "⬇️"
            lines.append(
                f"  {icon} {change['symbol']}: {change['morning_score']:.0f} → "
                f"{change['evening_score']:.0f} ({change['change']:+.0f})"
            )

    # Tomorrow's focus
    if briefing.alerts_for_tomorrow:
        lines.extend([
            "",
            "📌 TOMORROW'S FOCUS",
            "─" * 40,
        ])
        for alert in briefing.alerts_for_tomorrow:
            lines.append(f"  • {alert}")

    lines.append("")
    lines.append("═" * 60)

    return "\n".join(lines)
