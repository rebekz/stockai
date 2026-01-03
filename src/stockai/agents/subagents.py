"""Subagent Definitions.

Defines the 7 specialized subagents for the multi-agent trading system.
These subagents are used with the DeepAgents framework.
"""

from typing import Any

from stockai.agents.prompts import (
    MARKET_SCANNER_PROMPT,
    RESEARCH_AGENT_PROMPT,
    TECHNICAL_ANALYST_PROMPT,
    SENTIMENT_ANALYST_PROMPT,
    PORTFOLIO_MANAGER_PROMPT,
    RISK_MANAGER_PROMPT,
    TRADING_EXECUTION_PROMPT,
)
from stockai.agents.tools import get_agent_tools


def get_market_scanner_subagent() -> dict[str, Any]:
    """Get Market Scanner subagent definition."""
    return {
        "name": "market_scanner",
        "description": "Scans IDX market for trading opportunities based on price movements, volume spikes, sector rotations, and technical breakouts. Use this agent to discover new investment opportunities.",
        "system_prompt": MARKET_SCANNER_PROMPT,
        "tools": get_agent_tools("market_scanner"),
    }


def get_research_agent_subagent() -> dict[str, Any]:
    """Get Research Agent (Fundamental Analyst) subagent definition."""
    return {
        "name": "research_agent",
        "description": "Analyzes company fundamentals including financials, valuations, growth metrics, and competitive position for IDX stocks. Use this agent for fundamental analysis.",
        "system_prompt": RESEARCH_AGENT_PROMPT,
        "tools": get_agent_tools("research"),
    }


def get_technical_analyst_subagent() -> dict[str, Any]:
    """Get Technical Analyst subagent definition."""
    return {
        "name": "technical_analyst",
        "description": "Analyzes price charts, technical indicators, support/resistance levels, and chart patterns for IDX stocks. Use this agent for technical analysis.",
        "system_prompt": TECHNICAL_ANALYST_PROMPT,
        "tools": get_agent_tools("technical"),
    }


def get_sentiment_analyst_subagent() -> dict[str, Any]:
    """Get Sentiment Analyst subagent definition."""
    return {
        "name": "sentiment_analyst",
        "description": "Analyzes news sentiment, social media buzz, and market mood for IDX stocks using Indonesian financial news sources. Use this agent for sentiment analysis.",
        "system_prompt": SENTIMENT_ANALYST_PROMPT,
        "tools": get_agent_tools("sentiment"),
    }


def get_portfolio_manager_subagent() -> dict[str, Any]:
    """Get Portfolio Manager subagent definition."""
    return {
        "name": "portfolio_manager",
        "description": "Optimizes portfolio allocation, position sizing, and rebalancing strategies based on risk-adjusted returns. Use this agent for position sizing and portfolio optimization.",
        "system_prompt": PORTFOLIO_MANAGER_PROMPT,
        "tools": get_agent_tools("portfolio"),
    }


def get_risk_manager_subagent() -> dict[str, Any]:
    """Get Risk Manager subagent definition."""
    return {
        "name": "risk_manager",
        "description": "Assesses investment risks, calculates risk metrics, and sets stop-loss levels for capital protection. Use this agent for risk assessment.",
        "system_prompt": RISK_MANAGER_PROMPT,
        "tools": get_agent_tools("risk"),
    }


def get_trading_execution_subagent() -> dict[str, Any]:
    """Get Trading Execution subagent definition."""
    return {
        "name": "trading_execution",
        "description": "Synthesizes all analysis into actionable trading signals with specific entry, exit, and position sizing recommendations. Use this agent to generate final trading signals.",
        "system_prompt": TRADING_EXECUTION_PROMPT,
        "tools": get_agent_tools("execution"),
    }


def get_all_subagents() -> list[dict[str, Any]]:
    """Get all subagent definitions.

    Returns:
        List of subagent definition dictionaries
    """
    return [
        get_market_scanner_subagent(),
        get_research_agent_subagent(),
        get_technical_analyst_subagent(),
        get_sentiment_analyst_subagent(),
        get_portfolio_manager_subagent(),
        get_risk_manager_subagent(),
        get_trading_execution_subagent(),
    ]


# Convenience dict for looking up subagents by name
SUBAGENT_REGISTRY = {
    "market_scanner": get_market_scanner_subagent,
    "research_agent": get_research_agent_subagent,
    "technical_analyst": get_technical_analyst_subagent,
    "sentiment_analyst": get_sentiment_analyst_subagent,
    "portfolio_manager": get_portfolio_manager_subagent,
    "risk_manager": get_risk_manager_subagent,
    "trading_execution": get_trading_execution_subagent,
}


def get_subagent(name: str) -> dict[str, Any] | None:
    """Get a specific subagent by name.

    Args:
        name: Subagent name

    Returns:
        Subagent definition or None if not found
    """
    factory = SUBAGENT_REGISTRY.get(name)
    if factory:
        return factory()
    return None
