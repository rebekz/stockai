"""Autopilot Trading Module.

Automated daily trading system that:
- Scans Indonesian stock indices (JII70, IDX30, LQ45)
- Generates buy/sell signals using multi-factor scoring
- Validates signals with 7-agent AI orchestrator
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
from stockai.autopilot.validator import (
    AIValidator,
    AIValidatorConfig,
    ValidationResult,
    create_validator,
)

__all__ = [
    # Core engine
    "AutopilotEngine",
    "AutopilotConfig",
    "IndexType",
    "PaperExecutor",
    # AI validation
    "AIValidator",
    "AIValidatorConfig",
    "ValidationResult",
    "create_validator",
    # Utilities
    "format_autopilot_result",
    "get_autopilot_history",
    "format_autopilot_history",
]
