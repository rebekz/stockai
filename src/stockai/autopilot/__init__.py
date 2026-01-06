"""Autopilot Trading Module.

Automated daily trading system that:
- Scans Indonesian stock indices (JII70, IDX30, LQ45)
- Generates buy/sell signals using multi-factor scoring
- Calculates position sizes with 2% risk rule
- Executes trades via paper trading
"""

from stockai.autopilot.engine import (
    AutopilotEngine,
    AutopilotConfig,
    IndexType,
    format_autopilot_result,
    get_autopilot_history,
    format_autopilot_history,
)
from stockai.autopilot.executor import PaperExecutor

__all__ = [
    "AutopilotEngine",
    "AutopilotConfig",
    "IndexType",
    "PaperExecutor",
    "format_autopilot_result",
    "get_autopilot_history",
    "format_autopilot_history",
]
