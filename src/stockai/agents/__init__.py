"""StockAI Multi-Agent Trading System.

A 7-agent system for Indonesian stock (IDX) analysis and trading signals,
built with LangGraph orchestration.

Agents:
- MarketScanner: Discovers trading opportunities
- ResearchAgent: Fundamental analysis
- TechnicalAnalyst: Chart/indicator analysis
- SentimentAnalyst: News sentiment analysis
- PortfolioManager: Position sizing and allocation
- RiskManager: Risk assessment and stop-losses
- TradingExecution: Final signal generation
- Orchestrator: Coordinates all agents
"""

from stockai.agents.config import AgentConfig, get_agent_config
from stockai.agents.orchestrator import (
    TradingOrchestrator,
    create_trading_orchestrator,
    run_trading_analysis,
)
from stockai.agents.subagents import get_all_subagents, get_subagent
from stockai.agents.tools import get_agent_tools

__all__ = [
    # Config
    "AgentConfig",
    "get_agent_config",
    # Orchestrator
    "TradingOrchestrator",
    "create_trading_orchestrator",
    "run_trading_analysis",
    # Subagents
    "get_all_subagents",
    "get_subagent",
    # Tools
    "get_agent_tools",
]
