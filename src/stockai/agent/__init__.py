"""StockAI Agent Engine.

Autonomous AI research agent for Indonesian stock market analysis.
Uses LangChain/LangGraph with Google Gemini for multi-agent orchestration.
"""

from stockai.agent.orchestrator import StockAIAgent, create_agent

__all__ = ["StockAIAgent", "create_agent"]
