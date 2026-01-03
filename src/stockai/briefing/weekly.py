"""Weekly Review Module.

Weekly review for passive investors (30-60 minutes):
- Performance vs benchmark
- Win rate and trade analysis
- Portfolio rebalancing suggestions
- Score trends and changes
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import pytz

import logging

logger = logging.getLogger(__name__)

TIMEZONE = pytz.timezone("Asia/Jakarta")


@dataclass
class TradeAnalysis:
    """Analysis of trading activity for the week."""

    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0  # Total wins / Total losses
    largest_win: dict[str, Any] = field(default_factory=dict)
    largest_loss: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeeklyReview:
    """Weekly performance review."""

    week_start: datetime
    week_end: datetime

    # Portfolio performance
    starting_value: float = 0
    ending_value: float = 0
    weekly_pnl: float = 0
    weekly_pnl_pct: float = 0

    # Benchmark comparison (IHSG)
    ihsg_return_pct: float = 0
    alpha: float = 0  # Outperformance vs IHSG

    # Cumulative performance
    total_return_pct: float = 0  # Since inception
    initial_capital: float = 0

    # Trade analysis
    trade_analysis: TradeAnalysis = field(default_factory=TradeAnalysis)

    # Position performance
    best_performers: list[dict[str, Any]] = field(default_factory=list)
    worst_performers: list[dict[str, Any]] = field(default_factory=list)

    # Score trends
    score_improvements: list[dict[str, Any]] = field(default_factory=list)
    score_declines: list[dict[str, Any]] = field(default_factory=list)

    # Rebalancing suggestions
    rebalance_needed: bool = False
    rebalance_suggestions: list[str] = field(default_factory=list)

    # Risk metrics
    max_drawdown_week: float = 0
    portfolio_volatility: float = 0

    # Goals progress
    weekly_goal_target: float = 0.5  # 0.5% weekly = ~26% annual
    on_track: bool = True

    # Key lessons
    lessons_learned: list[str] = field(default_factory=list)


def generate_weekly_review(
    portfolio: dict[str, Any],
    trades_this_week: list[dict[str, Any]] | None = None,
    ihsg_weekly_return: float = 0.0,
    score_history: dict[str, list[float]] | None = None,
) -> WeeklyReview:
    """Generate comprehensive weekly review.

    Args:
        portfolio: Current portfolio state
        trades_this_week: List of trades executed this week
        ihsg_weekly_return: IHSG performance this week (%)
        score_history: Daily scores for the week per symbol

    Returns:
        WeeklyReview with all analysis
    """
    now = datetime.now(TIMEZONE)
    week_start = now - timedelta(days=now.weekday())
    week_end = now

    review = WeeklyReview(
        week_start=week_start,
        week_end=week_end,
        ihsg_return_pct=ihsg_weekly_return,
    )

    # Calculate portfolio performance
    if portfolio:
        positions = portfolio.get("positions", {})
        cash = portfolio.get("cash", 0)

        positions_value = sum(
            p.get("current_value", p.get("shares", 0) * p.get("avg_price", 0))
            for p in positions.values()
        )

        review.ending_value = positions_value + cash
        review.initial_capital = portfolio.get("initial_capital", review.ending_value)

        # Weekly P&L (simplified)
        review.starting_value = portfolio.get("week_start_value", review.ending_value * 0.99)
        review.weekly_pnl = review.ending_value - review.starting_value
        review.weekly_pnl_pct = (review.weekly_pnl / review.starting_value * 100) if review.starting_value > 0 else 0

        # Total return since inception
        review.total_return_pct = ((review.ending_value - review.initial_capital) / review.initial_capital * 100) if review.initial_capital > 0 else 0

        # Alpha vs IHSG
        review.alpha = review.weekly_pnl_pct - ihsg_weekly_return

        # Position performance
        position_returns = []
        for symbol, pos in positions.items():
            current = pos.get("current_price", 0)
            avg = pos.get("avg_price", 0)
            if avg > 0:
                return_pct = (current - avg) / avg * 100
                position_returns.append({
                    "symbol": symbol,
                    "return_pct": round(return_pct, 2),
                    "current_price": current,
                    "avg_price": avg,
                })

        position_returns.sort(key=lambda x: x["return_pct"], reverse=True)
        review.best_performers = position_returns[:3]
        review.worst_performers = position_returns[-3:][::-1] if len(position_returns) >= 3 else []

    # Analyze trades
    if trades_this_week:
        analysis = TradeAnalysis()
        analysis.total_trades = len(trades_this_week)

        wins = []
        losses = []

        for trade in trades_this_week:
            action = trade.get("action", "").upper()
            if action == "BUY":
                analysis.buy_trades += 1
            elif action == "SELL":
                analysis.sell_trades += 1
                # Calculate profit/loss for sell trades
                pnl = trade.get("pnl", 0)
                if pnl > 0:
                    analysis.winning_trades += 1
                    wins.append(pnl)
                    if not analysis.largest_win or pnl > analysis.largest_win.get("pnl", 0):
                        analysis.largest_win = {"symbol": trade.get("symbol"), "pnl": pnl}
                elif pnl < 0:
                    analysis.losing_trades += 1
                    losses.append(abs(pnl))
                    if not analysis.largest_loss or abs(pnl) > analysis.largest_loss.get("pnl", 0):
                        analysis.largest_loss = {"symbol": trade.get("symbol"), "pnl": pnl}

        if analysis.sell_trades > 0:
            analysis.win_rate = (analysis.winning_trades / analysis.sell_trades) * 100

        if wins:
            analysis.avg_win = sum(wins) / len(wins)
        if losses:
            analysis.avg_loss = sum(losses) / len(losses)

        if sum(losses) > 0:
            analysis.profit_factor = sum(wins) / sum(losses)

        review.trade_analysis = analysis

    # Score trends
    if score_history:
        for symbol, scores in score_history.items():
            if len(scores) >= 2:
                change = scores[-1] - scores[0]
                if change >= 10:
                    review.score_improvements.append({
                        "symbol": symbol,
                        "start_score": scores[0],
                        "end_score": scores[-1],
                        "change": change,
                    })
                elif change <= -10:
                    review.score_declines.append({
                        "symbol": symbol,
                        "start_score": scores[0],
                        "end_score": scores[-1],
                        "change": change,
                    })

    # Check if on track for goals
    if review.weekly_pnl_pct >= review.weekly_goal_target:
        review.on_track = True
    else:
        review.on_track = False

    # Generate lessons learned
    lessons = []
    if review.trade_analysis.win_rate < 50:
        lessons.append("Win rate below 50% - review entry criteria and timing")
    if review.trade_analysis.avg_loss > review.trade_analysis.avg_win * 1.5:
        lessons.append("Average loss exceeds average win - tighten stop-losses")
    if review.alpha < -2:
        lessons.append("Underperforming IHSG - consider more diversification")
    if not trades_this_week:
        lessons.append("No trades this week - review if signals are being generated")

    review.lessons_learned = lessons

    return review


def format_weekly_review(review: WeeklyReview) -> str:
    """Format weekly review for display.

    Args:
        review: WeeklyReview to format

    Returns:
        Formatted string for CLI
    """
    lines = [
        "╔" + "═" * 58 + "╗",
        f"║  📊 WEEKLY REVIEW: {review.week_start.strftime('%d %b')} - {review.week_end.strftime('%d %b %Y')}",
        "╠" + "═" * 58 + "╣",
    ]

    # Performance summary
    pnl_icon = "📈" if review.weekly_pnl >= 0 else "📉"
    alpha_icon = "✅" if review.alpha >= 0 else "⚠️"

    lines.extend([
        f"║ {pnl_icon} PERFORMANCE SUMMARY",
        "║ " + "─" * 56,
        f"║   Portfolio Value:  Rp {review.ending_value:>15,.0f}",
        f"║   Weekly P&L:       Rp {review.weekly_pnl:>15,.0f} ({review.weekly_pnl_pct:+.2f}%)",
        f"║   Total Return:     {review.total_return_pct:>19.2f}% (since inception)",
        f"║",
        f"║ {alpha_icon} BENCHMARK COMPARISON",
        f"║   IHSG This Week:   {review.ihsg_return_pct:>19.2f}%",
        f"║   Your Alpha:       {review.alpha:>19.2f}%",
    ])

    # On track status
    if review.on_track:
        lines.append(f"║   Status:           {'✅ ON TRACK':>19}")
    else:
        lines.append(f"║   Status:           {'⚠️ BELOW TARGET':>19}")

    # Trade analysis
    ta = review.trade_analysis
    if ta.total_trades > 0:
        lines.extend([
            "║",
            "║ 📋 TRADE ANALYSIS",
            "║ " + "─" * 56,
            f"║   Total Trades:     {ta.total_trades:>19}",
            f"║   Buys / Sells:     {ta.buy_trades:>9} / {ta.sell_trades}",
            f"║   Win Rate:         {ta.win_rate:>18.1f}%",
        ])

        if ta.avg_win > 0:
            lines.append(f"║   Avg Win:          Rp {ta.avg_win:>15,.0f}")
        if ta.avg_loss > 0:
            lines.append(f"║   Avg Loss:         Rp {ta.avg_loss:>15,.0f}")
        if ta.profit_factor > 0:
            lines.append(f"║   Profit Factor:    {ta.profit_factor:>19.2f}")

        if ta.largest_win:
            lines.append(f"║   Biggest Win:      {ta.largest_win.get('symbol', 'N/A')} (+Rp {ta.largest_win.get('pnl', 0):,.0f})")
        if ta.largest_loss:
            lines.append(f"║   Biggest Loss:     {ta.largest_loss.get('symbol', 'N/A')} (Rp {ta.largest_loss.get('pnl', 0):,.0f})")
    else:
        lines.extend([
            "║",
            "║ 📋 No trades executed this week",
        ])

    # Best/Worst performers
    if review.best_performers:
        lines.extend([
            "║",
            "║ 🏆 BEST PERFORMERS",
        ])
        for pos in review.best_performers[:3]:
            if pos["return_pct"] > 0:
                lines.append(f"║   📈 {pos['symbol']:8} +{pos['return_pct']:>6.2f}%")

    if review.worst_performers:
        lines.extend([
            "║",
            "║ 📉 WORST PERFORMERS",
        ])
        for pos in review.worst_performers[:3]:
            if pos["return_pct"] < 0:
                lines.append(f"║   📉 {pos['symbol']:8} {pos['return_pct']:>6.2f}%")

    # Score changes
    if review.score_improvements or review.score_declines:
        lines.extend([
            "║",
            "║ 📊 SCORE TRENDS",
            "║ " + "─" * 56,
        ])
        for change in review.score_improvements[:3]:
            lines.append(f"║   ⬆️ {change['symbol']}: {change['start_score']:.0f} → {change['end_score']:.0f} (+{change['change']:.0f})")
        for change in review.score_declines[:3]:
            lines.append(f"║   ⬇️ {change['symbol']}: {change['start_score']:.0f} → {change['end_score']:.0f} ({change['change']:.0f})")

    # Lessons learned
    if review.lessons_learned:
        lines.extend([
            "║",
            "║ 💡 KEY TAKEAWAYS",
            "║ " + "─" * 56,
        ])
        for lesson in review.lessons_learned:
            lines.append(f"║   • {lesson}")

    # Rebalancing
    if review.rebalance_suggestions:
        lines.extend([
            "║",
            "║ ⚖️ REBALANCING NEEDED",
        ])
        for suggestion in review.rebalance_suggestions[:3]:
            lines.append(f"║   • {suggestion}")

    lines.append("╚" + "═" * 58 + "╝")

    return "\n".join(lines)
