"""Trading Scheduler.

Schedule automated trading tasks based on market hours.
"""

import asyncio
from datetime import datetime, time
from typing import Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import pytz

import logging

logger = logging.getLogger(__name__)


class MarketSession(Enum):
    """IDX market sessions."""
    PRE_MARKET = "pre_market"      # 8:00-9:00
    SESSION_1 = "session_1"         # 9:00-12:00
    LUNCH = "lunch"                 # 12:00-13:30
    SESSION_2 = "session_2"         # 13:30-16:00
    POST_MARKET = "post_market"     # 16:00-17:00
    CLOSED = "closed"


@dataclass
class ScheduledTask:
    """A scheduled trading task."""
    name: str
    func: Callable
    run_at: time
    days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    enabled: bool = True
    last_run: datetime | None = None


class TradingScheduler:
    """Scheduler for automated trading tasks.

    Example:
        scheduler = TradingScheduler(capital=10_000_000)

        # Add pre-market scan
        scheduler.add_task(
            name="morning_scan",
            func=scheduler.run_daily_scan,
            run_at=time(8, 30),
        )

        # Start scheduler
        await scheduler.run()
    """

    TIMEZONE = pytz.timezone("Asia/Jakarta")

    def __init__(
        self,
        capital: int = 10_000_000,
        index: str = "IDX30",
        holdings: list[str] | None = None,
    ):
        self.capital = capital
        self.index = index
        self.holdings = holdings or []
        self.tasks: list[ScheduledTask] = []
        self._running = False
        self._callbacks: list[Callable] = []

    def get_current_session(self) -> MarketSession:
        """Get current market session."""
        now = datetime.now(self.TIMEZONE)
        current_time = now.time()
        weekday = now.weekday()

        # Weekend
        if weekday >= 5:
            return MarketSession.CLOSED

        # Check time ranges
        if current_time < time(8, 0):
            return MarketSession.CLOSED
        elif current_time < time(9, 0):
            return MarketSession.PRE_MARKET
        elif current_time < time(12, 0):
            return MarketSession.SESSION_1
        elif current_time < time(13, 30):
            return MarketSession.LUNCH
        elif current_time < time(16, 0):
            return MarketSession.SESSION_2
        elif current_time < time(17, 0):
            return MarketSession.POST_MARKET
        else:
            return MarketSession.CLOSED

    def is_market_open(self) -> bool:
        """Check if market is currently open for trading."""
        session = self.get_current_session()
        return session in (MarketSession.SESSION_1, MarketSession.SESSION_2)

    def add_task(
        self,
        name: str,
        func: Callable,
        run_at: time,
        days: list[int] | None = None,
        **kwargs,
    ) -> None:
        """Add a scheduled task.

        Args:
            name: Task name
            func: Function to call
            run_at: Time to run (Jakarta timezone)
            days: Days to run (0=Mon, 4=Fri). Default: Mon-Fri
            **kwargs: Additional arguments to pass to func
        """
        task = ScheduledTask(
            name=name,
            func=func,
            run_at=run_at,
            days=days or [0, 1, 2, 3, 4],
            kwargs=kwargs,
        )
        self.tasks.append(task)
        logger.info(f"Added task: {name} at {run_at}")

    def add_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Add callback for task completion.

        Callback receives (task_name, result).
        """
        self._callbacks.append(callback)

    def setup_default_schedule(self) -> None:
        """Setup default trading schedule.

        - 8:30 AM: Morning scan and daily recommendations
        - 9:15 AM: Post-open check
        - 12:00 PM: Mid-day review
        - 15:45 PM: Pre-close signals
        - 16:15 PM: End of day summary
        """
        from stockai.automation.runner import AutomatedTrader

        trader = AutomatedTrader(
            capital=self.capital,
            index=self.index,
            holdings=self.holdings,
        )

        # Morning scan
        self.add_task(
            name="morning_scan",
            func=trader.run_daily_recommendations,
            run_at=time(8, 30),
            horizon="both",
        )

        # Post-open check
        self.add_task(
            name="post_open_check",
            func=trader.run_market_scan,
            run_at=time(9, 15),
        )

        # Mid-day review
        self.add_task(
            name="midday_review",
            func=trader.run_portfolio_check,
            run_at=time(12, 0),
        )

        # Pre-close signals
        self.add_task(
            name="preclose_signals",
            func=trader.run_quick_signals,
            run_at=time(15, 45),
        )

        # EOD summary
        self.add_task(
            name="eod_summary",
            func=trader.run_daily_summary,
            run_at=time(16, 15),
        )

        logger.info("Default schedule configured with 5 tasks")

    async def _execute_task(self, task: ScheduledTask) -> Any:
        """Execute a single task."""
        logger.info(f"Executing task: {task.name}")

        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)

            task.last_run = datetime.now(self.TIMEZONE)

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(task.name, result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            return result

        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
            raise

    async def run(self, check_interval: int = 60) -> None:
        """Run the scheduler.

        Args:
            check_interval: Seconds between schedule checks
        """
        self._running = True
        logger.info("Trading scheduler started")

        while self._running:
            now = datetime.now(self.TIMEZONE)
            current_time = now.time()
            weekday = now.weekday()

            for task in self.tasks:
                if not task.enabled:
                    continue

                # Check if task should run
                if weekday not in task.days:
                    continue

                # Check time (within 1 minute window)
                task_minute = task.run_at.hour * 60 + task.run_at.minute
                current_minute = current_time.hour * 60 + current_time.minute

                if task_minute == current_minute:
                    # Check if already ran today
                    if task.last_run and task.last_run.date() == now.date():
                        continue

                    await self._execute_task(task)

            await asyncio.sleep(check_interval)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        logger.info("Trading scheduler stopped")
