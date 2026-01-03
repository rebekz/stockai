"""StockAI Trading Automation.

Automated trading workflow with scheduling, notifications, and execution.
"""

from stockai.automation.scheduler import TradingScheduler
from stockai.automation.notifier import Notifier, TelegramNotifier, EmailNotifier
from stockai.automation.runner import AutomatedTrader

__all__ = [
    "TradingScheduler",
    "Notifier",
    "TelegramNotifier",
    "EmailNotifier",
    "AutomatedTrader",
]
