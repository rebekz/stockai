"""Agent Configuration.

Central configuration for the multi-agent trading system.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for the multi-agent system."""

    # Model settings
    model: str = "gemini/gemini-3-flash-preview"
    temperature: float = 0.3
    max_tokens: int = 4096
    thinking_level: str = "HIGH"  # Options: MINIMAL, LOW, MEDIUM, HIGH

    # Workflow settings
    max_iterations: int = 3  # Reflection loop limit
    parallel_execution: bool = True
    cache_ttl: int = 300  # 5 minutes for market data

    # Scoring weights for Trading Execution
    scoring_weights: dict[str, float] = field(default_factory=lambda: {
        "fundamental": 0.25,
        "technical": 0.30,
        "sentiment": 0.20,
        "portfolio_fit": 0.15,
        "risk": 0.10,
    })

    # Action thresholds
    action_thresholds: dict[str, float] = field(default_factory=lambda: {
        "strong_buy": 7.5,
        "buy": 6.0,
        "hold_upper": 5.5,
        "hold_lower": 4.5,
        "sell": 4.0,
        "strong_sell": 0.0,
    })

    # Risk parameters
    max_loss_per_trade: float = 0.02  # 2% of portfolio
    max_portfolio_drawdown: float = 0.15  # 15%
    position_stop_loss_range: tuple[float, float] = (0.05, 0.10)  # 5-10%
    sector_concentration_limit: float = 0.40  # 40%


_config: AgentConfig | None = None


def get_agent_config() -> AgentConfig:
    """Get singleton agent configuration."""
    global _config
    if _config is None:
        _config = AgentConfig()
    return _config


def set_agent_config(config: AgentConfig) -> None:
    """Set agent configuration."""
    global _config
    _config = config
