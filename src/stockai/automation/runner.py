"""Automated Trading Runner.

Executes trading strategies and generates recommendations automatically.
"""

import json
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pytz

from stockai.agents import TradingOrchestrator, create_trading_orchestrator
from stockai.automation.notifier import Notifier, TradingAlert
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingResult:
    """Result from an automated trading run."""
    timestamp: datetime
    task_name: str
    success: bool
    recommendations: list[dict] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "task_name": self.task_name,
            "success": self.success,
            "recommendations": self.recommendations,
            "signals": self.signals,
            "errors": self.errors,
        }


class AutomatedTrader:
    """Automated trading execution.

    Example:
        trader = AutomatedTrader(capital=10_000_000)

        # Run daily recommendations
        result = await trader.run_daily_recommendations(horizon="short")

        # With notifications
        notifier = TelegramNotifier(token, chat_id)
        trader = AutomatedTrader(capital=10_000_000, notifier=notifier)
        await trader.run_daily_recommendations()
    """

    TIMEZONE = pytz.timezone("Asia/Jakarta")

    def __init__(
        self,
        capital: int = 10_000_000,
        index: str = "IDX30",
        holdings: list[str] | None = None,
        notifier: Notifier | None = None,
        output_dir: str | Path | None = None,
    ):
        self.capital = capital
        self.index = index
        self.holdings = holdings or []
        self.notifier = notifier
        self.output_dir = Path(output_dir) if output_dir else Path.home() / "stockai_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._orchestrator: TradingOrchestrator | None = None

    @property
    def orchestrator(self) -> TradingOrchestrator:
        """Lazy-load orchestrator."""
        if self._orchestrator is None:
            self._orchestrator = create_trading_orchestrator()
        return self._orchestrator

    def _save_result(self, result: TradingResult) -> Path:
        """Save result to file."""
        filename = f"{result.task_name}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"Result saved to {filepath}")
        return filepath

    async def _notify(self, alert: TradingAlert) -> None:
        """Send notification if notifier configured."""
        if self.notifier:
            await self.notifier.send(alert)

    def run_market_scan(self) -> TradingResult:
        """Run market scanner for opportunities."""
        result = TradingResult(
            timestamp=datetime.now(self.TIMEZONE),
            task_name="market_scan",
            success=False,
        )

        try:
            query = f"Scan {self.index} for top 5 trading opportunities today. Look for volume spikes, breakouts, and momentum plays."

            output = self.orchestrator.run(query)
            result.raw_output = output.get("answer", "")
            result.success = True

            # Extract opportunities from response
            if "market_scan" in output:
                result.recommendations = output["market_scan"].get("opportunities", [])

            logger.info(f"Market scan complete: {len(result.recommendations)} opportunities")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Market scan failed: {e}")

        self._save_result(result)
        return result

    def run_daily_recommendations(self, horizon: str = "both") -> TradingResult:
        """Run daily portfolio recommendations.

        Args:
            horizon: "short", "long", or "both"
        """
        result = TradingResult(
            timestamp=datetime.now(self.TIMEZONE),
            task_name="daily_recommendations",
            success=False,
        )

        try:
            # Build query
            holdings_str = ", ".join(self.holdings) if self.holdings else "none"
            horizon_text = {
                "short": "short-term gains (1-2 weeks)",
                "long": "long-term investments (3-12 months)",
                "both": "both short-term and long-term opportunities",
            }.get(horizon, "both short-term and long-term")

            query = f"""
            I have Rp {self.capital:,} to invest. My current holdings are: {holdings_str}.

            Scan {self.index} and provide:
            1. Specific stocks to BUY with exact lot quantities and entry prices
            2. For my holdings, tell me which to HOLD or SELL
            3. Focus on {horizon_text}
            4. Include stop-loss levels and target prices
            5. Consider 1 lot = 100 shares minimum for IDX
            """

            output = self.orchestrator.run(query)
            result.raw_output = output.get("answer", "")
            result.success = True

            # Parse recommendations
            if "trading_signal" in output and output["trading_signal"]:
                signal = output["trading_signal"]
                if "recommendations" in signal:
                    result.recommendations = signal["recommendations"]
                if "signals" in signal:
                    result.signals = signal["signals"]

            logger.info(f"Daily recommendations complete: {len(result.recommendations)} picks")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Daily recommendations failed: {e}")

        self._save_result(result)
        return result

    def run_quick_signals(self, symbols: list[str] | None = None) -> TradingResult:
        """Get quick trading signals for specific stocks.

        Args:
            symbols: List of symbols to analyze (default: current holdings)
        """
        result = TradingResult(
            timestamp=datetime.now(self.TIMEZONE),
            task_name="quick_signals",
            success=False,
        )

        target_symbols = symbols or self.holdings
        if not target_symbols:
            result.errors.append("No symbols specified")
            return result

        try:
            for symbol in target_symbols:
                query = f"Quick signal for {symbol}: BUY/SELL/HOLD with entry, target, stop-loss"

                output = self.orchestrator.run(query, symbol=symbol)

                signal_data = {
                    "symbol": symbol,
                    "signal": output.get("final_recommendation", "HOLD"),
                    "analysis": output.get("answer", ""),
                    "score": output.get("composite_score"),
                }

                result.signals.append(signal_data)

            result.success = True
            logger.info(f"Quick signals complete: {len(result.signals)} signals")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Quick signals failed: {e}")

        self._save_result(result)
        return result

    def run_portfolio_check(self) -> TradingResult:
        """Check current portfolio for any needed actions."""
        result = TradingResult(
            timestamp=datetime.now(self.TIMEZONE),
            task_name="portfolio_check",
            success=False,
        )

        if not self.holdings:
            result.errors.append("No holdings to check")
            return result

        try:
            holdings_str = ", ".join(self.holdings)
            query = f"""
            Portfolio check for: {holdings_str}

            For each position:
            1. Current technical status (bullish/bearish/neutral)
            2. Any stop-loss triggers?
            3. Any take-profit targets hit?
            4. Recommended action: HOLD, ADD, TRIM, or EXIT
            """

            output = self.orchestrator.run(query)
            result.raw_output = output.get("answer", "")
            result.success = True

            logger.info("Portfolio check complete")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Portfolio check failed: {e}")

        self._save_result(result)
        return result

    def run_daily_summary(self) -> TradingResult:
        """Generate end-of-day summary."""
        result = TradingResult(
            timestamp=datetime.now(self.TIMEZONE),
            task_name="daily_summary",
            success=False,
        )

        try:
            holdings_str = ", ".join(self.holdings) if self.holdings else "none"
            query = f"""
            End of day summary:
            - Portfolio: {holdings_str}
            - Capital available: Rp {self.capital:,}

            Provide:
            1. Market overview for today
            2. How my holdings performed
            3. Key news/events that affected the market
            4. Outlook for tomorrow
            """

            output = self.orchestrator.run(query)
            result.raw_output = output.get("answer", "")
            result.success = True

            # Send notification
            if self.notifier:
                alert = TradingAlert(
                    title="Daily Summary",
                    message=result.raw_output[:500] + "..." if len(result.raw_output) > 500 else result.raw_output,
                    signal="ALERT",
                )
                import asyncio
                asyncio.create_task(self._notify(alert))

            logger.info("Daily summary complete")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Daily summary failed: {e}")

        self._save_result(result)
        return result


async def run_automated_trading(
    capital: int = 10_000_000,
    index: str = "IDX30",
    holdings: list[str] | None = None,
    telegram_token: str | None = None,
    telegram_chat_id: str | None = None,
) -> None:
    """Run full automated trading workflow.

    This is the main entry point for automation.

    Example:
        # Run with notifications
        await run_automated_trading(
            capital=10_000_000,
            holdings=["BBRI", "TLKM"],
            telegram_token="YOUR_BOT_TOKEN",
            telegram_chat_id="YOUR_CHAT_ID",
        )
    """
    from stockai.automation.scheduler import TradingScheduler
    from stockai.automation.notifier import TelegramNotifier

    # Setup notifier
    notifier = None
    if telegram_token and telegram_chat_id:
        notifier = TelegramNotifier(telegram_token, telegram_chat_id)
        if await notifier.test_connection():
            logger.info("Telegram notifications enabled")
        else:
            logger.warning("Telegram connection failed, continuing without notifications")
            notifier = None

    # Create scheduler
    scheduler = TradingScheduler(
        capital=capital,
        index=index,
        holdings=holdings,
    )

    # Setup default schedule
    scheduler.setup_default_schedule()

    # Add notification callback
    if notifier:
        def on_task_complete(task_name: str, result: TradingResult):
            if result.success and result.signals:
                for signal in result.signals:
                    alert = TradingAlert(
                        title=f"{task_name}: {signal.get('symbol', 'Update')}",
                        message=signal.get("analysis", ""),
                        signal=signal.get("signal", "ALERT"),
                        symbol=signal.get("symbol"),
                    )
                    import asyncio
                    asyncio.create_task(notifier.send(alert))

        scheduler.add_callback(on_task_complete)

    # Run scheduler
    logger.info("Starting automated trading...")
    await scheduler.run()
