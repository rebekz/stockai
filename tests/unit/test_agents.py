"""Unit Tests for Multi-Agent Trading System.

Tests the 7-agent trading system:
- Config and settings
- Subagent definitions
- Tools wrapper functions
- Orchestrator workflow
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestAgentConfig:
    """Test agent configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from stockai.agents.config import AgentConfig

        config = AgentConfig()

        assert config.model == "gemini/gemini-3-flash-preview"
        assert config.temperature == 0.3
        assert config.max_tokens == 4096
        assert config.max_iterations == 3
        assert config.parallel_execution is True

    def test_scoring_weights(self):
        """Test scoring weights sum to 1.0."""
        from stockai.agents.config import AgentConfig

        config = AgentConfig()
        total_weight = sum(config.scoring_weights.values())

        assert abs(total_weight - 1.0) < 0.01

    def test_action_thresholds(self):
        """Test action thresholds are properly ordered."""
        from stockai.agents.config import AgentConfig

        config = AgentConfig()
        thresholds = config.action_thresholds

        assert thresholds["strong_buy"] > thresholds["buy"]
        assert thresholds["buy"] > thresholds["hold_upper"]
        assert thresholds["hold_lower"] > thresholds["sell"]

    def test_get_agent_config_singleton(self):
        """Test singleton pattern for config."""
        from stockai.agents.config import get_agent_config, set_agent_config, AgentConfig

        # Get default
        config1 = get_agent_config()
        config2 = get_agent_config()

        assert config1 is config2

        # Reset with new config
        new_config = AgentConfig(temperature=0.5)
        set_agent_config(new_config)

        config3 = get_agent_config()
        assert config3.temperature == 0.5

        # Reset to default for other tests
        set_agent_config(None)


class TestSubagents:
    """Test subagent definitions."""

    def test_get_all_subagents(self):
        """Test getting all subagent definitions."""
        from stockai.agents.subagents import get_all_subagents

        agents = get_all_subagents()

        assert len(agents) == 7
        assert all("name" in agent for agent in agents)
        assert all("description" in agent for agent in agents)
        assert all("system_prompt" in agent for agent in agents)
        assert all("tools" in agent for agent in agents)

    def test_subagent_names(self):
        """Test all expected agents are present."""
        from stockai.agents.subagents import get_all_subagents

        agents = get_all_subagents()
        names = {agent["name"] for agent in agents}

        expected = {
            "market_scanner",
            "research_agent",
            "technical_analyst",
            "sentiment_analyst",
            "portfolio_manager",
            "risk_manager",
            "trading_execution",
        }

        assert names == expected

    def test_get_subagent_by_name(self):
        """Test getting specific subagent."""
        from stockai.agents.subagents import get_subagent

        agent = get_subagent("market_scanner")

        assert agent is not None
        assert agent["name"] == "market_scanner"
        assert len(agent["tools"]) > 0

    def test_get_subagent_not_found(self):
        """Test getting non-existent subagent."""
        from stockai.agents.subagents import get_subagent

        agent = get_subagent("nonexistent_agent")

        assert agent is None

    def test_market_scanner_subagent(self):
        """Test Market Scanner agent definition."""
        from stockai.agents.subagents import get_market_scanner_subagent

        agent = get_market_scanner_subagent()

        assert agent["name"] == "market_scanner"
        assert "IDX" in agent["description"]
        assert "opportunities" in agent["description"].lower()
        assert len(agent["tools"]) >= 4

    def test_research_agent_subagent(self):
        """Test Research Agent definition."""
        from stockai.agents.subagents import get_research_agent_subagent

        agent = get_research_agent_subagent()

        assert agent["name"] == "research_agent"
        assert "fundamental" in agent["description"].lower()
        assert len(agent["tools"]) >= 4

    def test_trading_execution_subagent(self):
        """Test Trading Execution agent definition."""
        from stockai.agents.subagents import get_trading_execution_subagent

        agent = get_trading_execution_subagent()

        assert agent["name"] == "trading_execution"
        assert "signal" in agent["description"].lower()


class TestAgentTools:
    """Test agent tool wrapper functions."""

    def test_all_tools_defined(self):
        """Test all tools are defined."""
        from stockai.agents.tools import ALL_TOOLS

        assert len(ALL_TOOLS) >= 10

    def test_get_agent_tools_all(self):
        """Test getting all tools."""
        from stockai.agents.tools import get_agent_tools, ALL_TOOLS

        tools = get_agent_tools(None)

        assert len(tools) == len(ALL_TOOLS)

    def test_get_agent_tools_market_scanner(self):
        """Test getting Market Scanner tools."""
        from stockai.agents.tools import get_agent_tools

        tools = get_agent_tools("market_scanner")

        tool_names = {tool.name for tool in tools}
        assert "get_lq45_list" in tool_names
        assert "get_idx30_list" in tool_names

    def test_get_agent_tools_research(self):
        """Test getting Research Agent tools."""
        from stockai.agents.tools import get_agent_tools

        tools = get_agent_tools("research")

        tool_names = {tool.name for tool in tools}
        assert "get_stock_info" in tool_names
        assert "get_financials" in tool_names

    def test_get_agent_tools_technical(self):
        """Test getting Technical Analyst tools."""
        from stockai.agents.tools import get_agent_tools

        tools = get_agent_tools("technical")

        tool_names = {tool.name for tool in tools}
        assert "generate_features" in tool_names
        assert "get_price_history" in tool_names

    def test_get_agent_tools_sentiment(self):
        """Test getting Sentiment Analyst tools."""
        from stockai.agents.tools import get_agent_tools

        tools = get_agent_tools("sentiment")

        tool_names = {tool.name for tool in tools}
        assert "fetch_stock_news" in tool_names
        assert "analyze_sentiment" in tool_names

    def test_get_agent_tools_unknown(self):
        """Test getting tools for unknown agent type."""
        from stockai.agents.tools import get_agent_tools, ALL_TOOLS

        tools = get_agent_tools("unknown_type")

        # Should return all tools for unknown type
        assert len(tools) == len(ALL_TOOLS)


class TestAgentPrompts:
    """Test agent prompts."""

    def test_orchestrator_prompt(self):
        """Test orchestrator prompt is defined."""
        from stockai.agents.prompts import ORCHESTRATOR_PROMPT

        assert ORCHESTRATOR_PROMPT is not None
        assert len(ORCHESTRATOR_PROMPT) > 100
        assert "market_scanner" in ORCHESTRATOR_PROMPT
        assert "trading_execution" in ORCHESTRATOR_PROMPT

    def test_market_scanner_prompt(self):
        """Test Market Scanner prompt."""
        from stockai.agents.prompts import MARKET_SCANNER_PROMPT

        assert "volume spike" in MARKET_SCANNER_PROMPT.lower()
        assert "breakout" in MARKET_SCANNER_PROMPT.lower()

    def test_research_agent_prompt(self):
        """Test Research Agent prompt."""
        from stockai.agents.prompts import RESEARCH_AGENT_PROMPT

        assert "fundamental" in RESEARCH_AGENT_PROMPT.lower()
        assert "p/e" in RESEARCH_AGENT_PROMPT.lower()

    def test_technical_analyst_prompt(self):
        """Test Technical Analyst prompt."""
        from stockai.agents.prompts import TECHNICAL_ANALYST_PROMPT

        assert "rsi" in TECHNICAL_ANALYST_PROMPT.lower()
        assert "macd" in TECHNICAL_ANALYST_PROMPT.lower()
        assert "support" in TECHNICAL_ANALYST_PROMPT.lower()

    def test_sentiment_analyst_prompt(self):
        """Test Sentiment Analyst prompt."""
        from stockai.agents.prompts import SENTIMENT_ANALYST_PROMPT

        assert "sentiment" in SENTIMENT_ANALYST_PROMPT.lower()
        assert "news" in SENTIMENT_ANALYST_PROMPT.lower()

    def test_portfolio_manager_prompt(self):
        """Test Portfolio Manager prompt."""
        from stockai.agents.prompts import PORTFOLIO_MANAGER_PROMPT

        assert "position" in PORTFOLIO_MANAGER_PROMPT.lower()
        assert "allocation" in PORTFOLIO_MANAGER_PROMPT.lower()

    def test_risk_manager_prompt(self):
        """Test Risk Manager prompt."""
        from stockai.agents.prompts import RISK_MANAGER_PROMPT

        assert "risk" in RISK_MANAGER_PROMPT.lower()
        assert "stop-loss" in RISK_MANAGER_PROMPT.lower()

    def test_trading_execution_prompt(self):
        """Test Trading Execution prompt."""
        from stockai.agents.prompts import TRADING_EXECUTION_PROMPT

        assert "buy" in TRADING_EXECUTION_PROMPT.lower()
        assert "sell" in TRADING_EXECUTION_PROMPT.lower()
        assert "entry" in TRADING_EXECUTION_PROMPT.lower()


class TestTradingState:
    """Test trading workflow state."""

    def test_create_initial_trading_state(self):
        """Test creating initial trading state."""
        from stockai.agents.orchestrator import create_initial_trading_state

        state = create_initial_trading_state("Analyze BBCA", symbol="BBCA")

        assert state["query"] == "Analyze BBCA"
        assert state["symbol"] == "BBCA"
        assert state["phase"] == "understand"
        assert state["iteration"] == 0
        assert state["answer"] is None
        assert state["started_at"] is not None

    def test_initial_state_no_symbol(self):
        """Test creating state without symbol."""
        from stockai.agents.orchestrator import create_initial_trading_state

        state = create_initial_trading_state("Scan for opportunities")

        assert state["query"] == "Scan for opportunities"
        assert state["symbol"] is None


class TestTradingOrchestrator:
    """Test trading orchestrator."""

    @patch("stockai.agents.orchestrator.ChatGoogleGenerativeAI")
    @patch("stockai.agents.orchestrator.get_settings")
    def test_orchestrator_initialization(self, mock_settings, mock_llm):
        """Test orchestrator initializes correctly."""
        mock_settings.return_value = Mock(google_api_key="test_key")
        mock_llm.return_value = Mock()

        from stockai.agents.orchestrator import TradingOrchestrator

        orchestrator = TradingOrchestrator()

        assert orchestrator.config is not None
        assert orchestrator.llm is not None
        assert len(orchestrator.agents) == 7

    @patch("stockai.agents.orchestrator.ChatGoogleGenerativeAI")
    @patch("stockai.agents.orchestrator.get_settings")
    def test_orchestrator_agents_initialized(self, mock_settings, mock_llm):
        """Test all agents are initialized."""
        mock_settings.return_value = Mock(google_api_key="test_key")
        mock_llm.return_value = Mock()

        from stockai.agents.orchestrator import TradingOrchestrator

        orchestrator = TradingOrchestrator()

        expected_agents = {
            "market_scanner",
            "research_agent",
            "technical_analyst",
            "sentiment_analyst",
            "portfolio_manager",
            "risk_manager",
            "trading_execution",
        }

        assert set(orchestrator.agents.keys()) == expected_agents

    @patch("stockai.agents.orchestrator.ChatGoogleGenerativeAI")
    @patch("stockai.agents.orchestrator.get_settings")
    def test_extract_score(self, mock_settings, mock_llm):
        """Test score extraction from analysis text."""
        mock_settings.return_value = Mock(google_api_key="test_key")
        mock_llm.return_value = Mock()

        from stockai.agents.orchestrator import TradingOrchestrator

        orchestrator = TradingOrchestrator()

        # Test various formats
        assert orchestrator._extract_score("SCORE: 7.5/10") == 7.5
        assert orchestrator._extract_score("Score: 8/10") == 8.0
        assert orchestrator._extract_score("Technical Score: 6.5") == 6.5
        assert orchestrator._extract_score("No score here") is None

    @patch("stockai.agents.orchestrator.ChatGoogleGenerativeAI")
    @patch("stockai.agents.orchestrator.get_settings")
    def test_calculate_composite_score(self, mock_settings, mock_llm):
        """Test composite score calculation."""
        mock_settings.return_value = Mock(google_api_key="test_key")
        mock_llm.return_value = Mock()

        from stockai.agents.orchestrator import TradingOrchestrator, TradingState

        orchestrator = TradingOrchestrator()

        # Create mock state with agent results
        state = TradingState(
            query="test",
            symbol="BBCA",
            intent="analyze",
            messages=[],
            phase="execute",
            iteration=0,
            market_scan=None,
            fundamental_analysis={"score": 7.0},
            technical_analysis={"score": 8.0},
            sentiment_analysis={"score": 6.0},
            portfolio_recommendation={"score": 7.5},
            risk_assessment={"score": 8.5},
            trading_signal=None,
            composite_score=None,
            final_recommendation=None,
            answer=None,
            started_at=datetime.utcnow().isoformat(),
            completed_at=None,
        )

        score = orchestrator._calculate_composite_score(state)

        # Score should be weighted average
        assert 6.0 <= score <= 9.0

    @patch("stockai.agents.orchestrator.ChatGoogleGenerativeAI")
    @patch("stockai.agents.orchestrator.get_settings")
    def test_count_agents_executed(self, mock_settings, mock_llm):
        """Test counting executed agents."""
        mock_settings.return_value = Mock(google_api_key="test_key")
        mock_llm.return_value = Mock()

        from stockai.agents.orchestrator import TradingOrchestrator, TradingState

        orchestrator = TradingOrchestrator()

        state = TradingState(
            query="test",
            symbol="BBCA",
            intent="analyze",
            messages=[],
            phase="complete",
            iteration=0,
            market_scan={"agent_name": "market_scanner"},
            fundamental_analysis={"agent_name": "research_agent"},
            technical_analysis=None,
            sentiment_analysis={"agent_name": "sentiment_analyst"},
            portfolio_recommendation=None,
            risk_assessment=None,
            trading_signal=None,
            composite_score=None,
            final_recommendation=None,
            answer=None,
            started_at=datetime.utcnow().isoformat(),
            completed_at=None,
        )

        executed = orchestrator._count_agents_executed(state)

        assert len(executed) == 3
        assert "market_scanner" in executed
        assert "research_agent" in executed
        assert "sentiment_analyst" in executed


class TestFactoryFunctions:
    """Test factory functions."""

    @patch("stockai.agents.orchestrator.ChatGoogleGenerativeAI")
    @patch("stockai.agents.orchestrator.get_settings")
    def test_create_trading_orchestrator(self, mock_settings, mock_llm):
        """Test creating orchestrator via factory."""
        mock_settings.return_value = Mock(google_api_key="test_key")
        mock_llm.return_value = Mock()

        from stockai.agents import create_trading_orchestrator

        orchestrator = create_trading_orchestrator()

        assert orchestrator is not None
        assert len(orchestrator.agents) == 7

    @patch("stockai.agents.orchestrator.ChatGoogleGenerativeAI")
    @patch("stockai.agents.orchestrator.get_settings")
    def test_create_trading_orchestrator_with_model(self, mock_settings, mock_llm):
        """Test creating orchestrator with custom model."""
        mock_settings.return_value = Mock(google_api_key="test_key")
        mock_llm.return_value = Mock()

        from stockai.agents import create_trading_orchestrator

        orchestrator = create_trading_orchestrator(model_name="gemini-pro")

        assert orchestrator.config.model == "gemini-pro"


class TestPackageExports:
    """Test package exports."""

    def test_agents_package_exports(self):
        """Test all expected exports are available."""
        from stockai.agents import (
            AgentConfig,
            get_agent_config,
            TradingOrchestrator,
            create_trading_orchestrator,
            run_trading_analysis,
            get_all_subagents,
            get_subagent,
            get_agent_tools,
        )

        # Just verify imports work
        assert AgentConfig is not None
        assert get_agent_config is not None
        assert TradingOrchestrator is not None
        assert create_trading_orchestrator is not None
        assert run_trading_analysis is not None
        assert get_all_subagents is not None
        assert get_subagent is not None
        assert get_agent_tools is not None
