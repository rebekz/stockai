"""Multi-Agent Trading System Orchestrator.

Coordinates 7 specialized agents for comprehensive stock analysis:
- Market Scanner: Discovers trading opportunities
- Research Agent: Fundamental analysis
- Technical Analyst: Chart/indicator analysis
- Sentiment Analyst: News sentiment analysis
- Portfolio Manager: Position sizing and allocation
- Risk Manager: Risk assessment and stop-losses
- Trading Execution: Final signal generation

Uses LangGraph for workflow orchestration with parallel and sequential execution.
"""

import json
import logging
import re
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from stockai.agents.config import AgentConfig, get_agent_config
from stockai.agents.prompts import ORCHESTRATOR_PROMPT
from stockai.agents.subagents import get_all_subagents, get_subagent
from stockai.agents.tools import get_agent_tools
from stockai.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# State Definitions
# =============================================================================


class AgentResult(TypedDict):
    """Result from a single agent execution."""

    agent_name: str
    score: float | None
    analysis: str
    raw_output: str
    executed_at: str


class TradingState(TypedDict):
    """State for the multi-agent trading workflow."""

    # User input
    query: str
    symbol: str | None
    intent: str | None  # scan, analyze, recommend, risk, portfolio

    # Messages for LLM context
    messages: Annotated[list[BaseMessage], add_messages]

    # Workflow tracking
    phase: str  # understand, scan, research, evaluate, execute
    iteration: int

    # Agent results
    market_scan: AgentResult | None
    fundamental_analysis: AgentResult | None
    technical_analysis: AgentResult | None
    sentiment_analysis: AgentResult | None
    portfolio_recommendation: AgentResult | None
    risk_assessment: AgentResult | None
    trading_signal: AgentResult | None

    # Final output
    composite_score: float | None
    final_recommendation: str | None
    answer: str | None

    # Metadata
    started_at: str
    completed_at: str | None


def create_initial_trading_state(query: str, symbol: str | None = None) -> TradingState:
    """Create initial state for trading workflow.

    Args:
        query: User's query
        symbol: Optional stock symbol

    Returns:
        Initial TradingState
    """
    return TradingState(
        query=query,
        symbol=symbol,
        intent=None,
        messages=[],
        phase="understand",
        iteration=0,
        market_scan=None,
        fundamental_analysis=None,
        technical_analysis=None,
        sentiment_analysis=None,
        portfolio_recommendation=None,
        risk_assessment=None,
        trading_signal=None,
        composite_score=None,
        final_recommendation=None,
        answer=None,
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
    )


# =============================================================================
# Multi-Agent Orchestrator
# =============================================================================


class TradingOrchestrator:
    """Orchestrates multiple specialized agents for stock trading analysis.

    This orchestrator implements the workflow:
    1. Understand: Parse user intent
    2. Scan (optional): Find opportunities with Market Scanner
    3. Research: Parallel fundamental, technical, sentiment analysis
    4. Evaluate: Portfolio fit and risk assessment
    5. Execute: Generate trading signal

    Attributes:
        config: Agent configuration settings
        llm: Language model for orchestration
        agents: Dictionary of agent instances
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        model_name: str | None = None,
    ):
        """Initialize the trading orchestrator.

        Args:
            config: Agent configuration (optional)
            model_name: Override model name
        """
        self.config = config or get_agent_config()

        if model_name:
            self.config.model = model_name

        # Initialize LLM
        self.llm = self._create_llm()

        # Initialize specialized agents
        self.agents = self._initialize_agents()

        # Build workflow
        self.workflow = self._build_workflow()

        logger.info(f"TradingOrchestrator initialized with {len(self.agents)} agents")

    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Create the orchestrator LLM."""
        settings = get_settings()

        # Extract model ID from config (e.g., "gemini/gemini-2.0-flash" -> "gemini-2.0-flash")
        model_id = self.config.model.split("/")[-1] if "/" in self.config.model else self.config.model

        # Build generation config with thinking enabled for supported models
        generation_config = {}
        if "preview" in model_id or "3-" in model_id:
            # Enable thinking mode for Gemini 3.x models
            # Options: MINIMAL, LOW, MEDIUM, HIGH
            generation_config["thinking_config"] = {
                "thinking_level": self.config.thinking_level
            }

        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=settings.google_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            convert_system_message_to_human=True,
            model_kwargs=generation_config if generation_config else None,
        )

    def _initialize_agents(self) -> dict[str, Any]:
        """Initialize all specialized agents.

        Returns:
            Dictionary mapping agent names to their configurations
        """
        agents = {}

        for subagent_def in get_all_subagents():
            name = subagent_def["name"]
            agents[name] = {
                "definition": subagent_def,
                "tools": subagent_def["tools"],
                "prompt": subagent_def["system_prompt"],
            }
            logger.debug(f"Initialized agent: {name}")

        return agents

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for agent orchestration.

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(TradingState)

        # Add nodes for each phase
        workflow.add_node("understand", self._understand_phase)
        workflow.add_node("scan", self._scan_phase)
        workflow.add_node("research_parallel", self._research_phase)
        workflow.add_node("evaluate", self._evaluate_phase)
        workflow.add_node("execute", self._execute_phase)
        workflow.add_node("synthesize", self._synthesize_phase)

        # Entry point
        workflow.set_entry_point("understand")

        # Conditional routing after understanding
        workflow.add_conditional_edges(
            "understand",
            self._route_after_understand,
            {
                "scan": "scan",
                "research": "research_parallel",
                "portfolio": "evaluate",
                "answer_only": "synthesize",
            },
        )

        # After scan, go to research
        workflow.add_edge("scan", "research_parallel")

        # After research, go to evaluate
        workflow.add_edge("research_parallel", "evaluate")

        # After evaluate, check if we need trading signal
        workflow.add_conditional_edges(
            "evaluate",
            self._should_execute,
            {
                "execute": "execute",
                "synthesize": "synthesize",
            },
        )

        # After execute, synthesize
        workflow.add_edge("execute", "synthesize")

        # End
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    # =========================================================================
    # Workflow Phases
    # =========================================================================

    def _understand_phase(self, state: TradingState) -> dict:
        """Parse user intent and determine workflow path.

        Args:
            state: Current workflow state

        Returns:
            State updates with intent
        """
        logger.info("Understand phase: Parsing user intent")

        query = state["query"].lower()

        # Determine intent from query
        if any(word in query for word in ["scan", "opportunities", "find", "discover", "screen"]):
            intent = "scan"
        elif any(word in query for word in ["buy", "sell", "should i", "recommend", "signal"]):
            intent = "recommend"
        elif any(word in query for word in ["risk", "stop-loss", "drawdown", "volatility"]):
            intent = "risk"
        elif any(word in query for word in ["portfolio", "allocation", "position size", "diversif"]):
            intent = "portfolio"
        elif any(word in query for word in ["analyze", "analysis", "research", "fundamentals", "technical"]):
            intent = "analyze"
        else:
            intent = "general"

        # Extract symbol if present
        symbol = state.get("symbol")
        if not symbol:
            # Try to extract from query (simple pattern: 4 uppercase letters)
            symbol_match = re.search(r"\b([A-Z]{4})\b", state["query"])
            if symbol_match:
                symbol = symbol_match.group(1)

        logger.info(f"Intent detected: {intent}, Symbol: {symbol}")

        return {
            "intent": intent,
            "symbol": symbol,
            "phase": "understand",
        }

    def _route_after_understand(self, state: TradingState) -> str:
        """Route to appropriate workflow based on intent."""
        intent = state.get("intent", "general")

        if intent == "scan":
            return "scan"
        elif intent in ["analyze", "recommend", "risk"]:
            return "research"
        elif intent == "portfolio":
            return "portfolio"
        else:
            return "answer_only"

    def _scan_phase(self, state: TradingState) -> dict:
        """Execute market scanning with Market Scanner agent."""
        logger.info("Scan phase: Running Market Scanner")

        result = self._run_agent(
            "market_scanner",
            state["query"],
            state.get("symbol"),
        )

        return {
            "market_scan": result,
            "phase": "scan",
        }

    def _research_phase(self, state: TradingState) -> dict:
        """Execute parallel research with multiple agents.

        Runs fundamental, technical, and sentiment analysis in parallel.
        """
        logger.info("Research phase: Running parallel analysis")

        symbol = state.get("symbol")
        if not symbol:
            logger.warning("No symbol provided for research")
            return {"phase": "research"}

        # Run all research agents
        # Note: In production, these would run in parallel using asyncio
        fundamental = self._run_agent("research_agent", state["query"], symbol)
        technical = self._run_agent("technical_analyst", state["query"], symbol)
        sentiment = self._run_agent("sentiment_analyst", state["query"], symbol)

        return {
            "fundamental_analysis": fundamental,
            "technical_analysis": technical,
            "sentiment_analysis": sentiment,
            "phase": "research",
        }

    def _evaluate_phase(self, state: TradingState) -> dict:
        """Execute portfolio and risk evaluation."""
        logger.info("Evaluate phase: Portfolio fit and risk assessment")

        symbol = state.get("symbol")

        # Run portfolio and risk agents
        portfolio = self._run_agent("portfolio_manager", state["query"], symbol)
        risk = self._run_agent("risk_manager", state["query"], symbol)

        return {
            "portfolio_recommendation": portfolio,
            "risk_assessment": risk,
            "phase": "evaluate",
        }

    def _should_execute(self, state: TradingState) -> str:
        """Determine if trading signal should be generated."""
        intent = state.get("intent", "general")

        # Generate trading signal for recommend and analyze intents
        if intent in ["recommend", "analyze"]:
            return "execute"
        return "synthesize"

    def _execute_phase(self, state: TradingState) -> dict:
        """Generate final trading signal."""
        logger.info("Execute phase: Generating trading signal")

        symbol = state.get("symbol")

        # Run trading execution agent
        signal = self._run_agent("trading_execution", state["query"], symbol)

        # Calculate composite score
        composite_score = self._calculate_composite_score(state)

        # Determine recommendation
        thresholds = self.config.action_thresholds
        if composite_score >= thresholds["strong_buy"]:
            recommendation = "STRONG BUY"
        elif composite_score >= thresholds["buy"]:
            recommendation = "BUY"
        elif composite_score >= thresholds["hold_upper"]:
            recommendation = "HOLD"
        elif composite_score >= thresholds["sell"]:
            recommendation = "SELL"
        else:
            recommendation = "STRONG SELL"

        return {
            "trading_signal": signal,
            "composite_score": composite_score,
            "final_recommendation": recommendation,
            "phase": "execute",
        }

    def _synthesize_phase(self, state: TradingState) -> dict:
        """Synthesize all agent outputs into final answer."""
        logger.info("Synthesize phase: Generating final answer")

        # Build comprehensive answer from all agent results
        sections = []

        if state.get("market_scan"):
            sections.append(f"## Market Scan\n{state['market_scan'].get('analysis', 'N/A')}")

        if state.get("fundamental_analysis"):
            sections.append(f"## Fundamental Analysis\n{state['fundamental_analysis'].get('analysis', 'N/A')}")

        if state.get("technical_analysis"):
            sections.append(f"## Technical Analysis\n{state['technical_analysis'].get('analysis', 'N/A')}")

        if state.get("sentiment_analysis"):
            sections.append(f"## Sentiment Analysis\n{state['sentiment_analysis'].get('analysis', 'N/A')}")

        if state.get("portfolio_recommendation"):
            sections.append(f"## Portfolio Recommendation\n{state['portfolio_recommendation'].get('analysis', 'N/A')}")

        if state.get("risk_assessment"):
            sections.append(f"## Risk Assessment\n{state['risk_assessment'].get('analysis', 'N/A')}")

        if state.get("trading_signal"):
            sections.append(f"## Trading Signal\n{state['trading_signal'].get('analysis', 'N/A')}")

        # Add final recommendation if available
        if state.get("final_recommendation"):
            rec = state["final_recommendation"]
            score = state.get("composite_score", 0)
            sections.insert(0, f"# Recommendation: {rec}\n**Composite Score: {score:.1f}/10**\n")

        answer = "\n\n".join(sections) if sections else "No analysis available."

        return {
            "answer": answer,
            "phase": "complete",
            "completed_at": datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # Agent Execution
    # =========================================================================

    def _run_agent(
        self,
        agent_name: str,
        query: str,
        symbol: str | None = None,
    ) -> AgentResult:
        """Run a specific agent.

        Args:
            agent_name: Name of agent to run
            query: User query
            symbol: Stock symbol (optional)

        Returns:
            AgentResult with analysis
        """
        logger.info(f"Running agent: {agent_name}")

        agent = self.agents.get(agent_name)
        if not agent:
            logger.warning(f"Agent not found: {agent_name}")
            return AgentResult(
                agent_name=agent_name,
                score=None,
                analysis="Agent not available",
                raw_output="",
                executed_at=datetime.utcnow().isoformat(),
            )

        try:
            # Build prompt for agent
            system_prompt = agent["prompt"]
            tools = agent["tools"]

            # Build context message
            context = f"Query: {query}"
            if symbol:
                context += f"\nStock Symbol: {symbol}"

            # Execute tools to gather data
            tool_outputs = []
            for tool in tools[:5]:  # Limit tool calls
                try:
                    if hasattr(tool, "invoke"):
                        if symbol:
                            result = tool.invoke({"symbol": symbol})
                        else:
                            result = tool.invoke({})
                        tool_outputs.append(f"**{tool.name}**: {json.dumps(result, default=str)[:500]}")
                except Exception as e:
                    logger.debug(f"Tool {getattr(tool, 'name', 'unknown')} failed: {e}")

            # Build analysis prompt
            analysis_prompt = f"""Based on the following data, provide your analysis:

{chr(10).join(tool_outputs) if tool_outputs else 'No data available.'}

{context}

Provide your analysis following your standard output format. Include a score from 1-10 at the end."""

            # Get analysis from LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=analysis_prompt),
            ]

            response = self.llm.invoke(messages)

            # Handle Gemini 3 thinking mode response format (returns list)
            content = response.content
            if isinstance(content, list):
                # Extract text from structured response
                analysis = ""
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        analysis += item["text"]
                    elif isinstance(item, str):
                        analysis += item
            else:
                analysis = content

            # Extract score from analysis
            score = self._extract_score(analysis)

            return AgentResult(
                agent_name=agent_name,
                score=score,
                analysis=analysis,
                raw_output=str(tool_outputs),
                executed_at=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            return AgentResult(
                agent_name=agent_name,
                score=None,
                analysis=f"Analysis failed: {str(e)}",
                raw_output="",
                executed_at=datetime.utcnow().isoformat(),
            )

    def _extract_score(self, text: str) -> float | None:
        """Extract score from agent analysis text."""
        # Look for patterns like "SCORE: 7.5/10" or "7/10" or "Score: 8"
        patterns = [
            r"(?:SCORE|Score)[:\s]+(\d+(?:\.\d+)?)\s*/\s*10",
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"(?:SCORE|Score)[:\s]+(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def _calculate_composite_score(self, state: TradingState) -> float:
        """Calculate weighted composite score from all agents."""
        weights = self.config.scoring_weights

        scores = []
        total_weight = 0

        # Collect scores with weights
        agent_score_map = {
            "fundamental": state.get("fundamental_analysis", {}).get("score"),
            "technical": state.get("technical_analysis", {}).get("score"),
            "sentiment": state.get("sentiment_analysis", {}).get("score"),
            "portfolio_fit": state.get("portfolio_recommendation", {}).get("score"),
            "risk": state.get("risk_assessment", {}).get("score"),
        }

        for key, score in agent_score_map.items():
            if score is not None:
                weight = weights.get(key, 0)
                scores.append(score * weight)
                total_weight += weight

        if total_weight > 0:
            return sum(scores) / total_weight
        return 5.0  # Neutral default

    # =========================================================================
    # Public Interface
    # =========================================================================

    def run(
        self,
        query: str,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Run the multi-agent trading analysis.

        Args:
            query: User's query
            symbol: Optional stock symbol

        Returns:
            Analysis results with answer and metadata
        """
        logger.info(f"TradingOrchestrator starting: {query}")

        # Create initial state
        state = create_initial_trading_state(query, symbol)

        try:
            # Run workflow
            final_state = self.workflow.invoke(state)

            return {
                "success": True,
                "query": query,
                "symbol": final_state.get("symbol"),
                "intent": final_state.get("intent"),
                "recommendation": final_state.get("final_recommendation"),
                "composite_score": final_state.get("composite_score"),
                "answer": final_state.get("answer"),
                "agents_executed": self._count_agents_executed(final_state),
                "started_at": final_state.get("started_at"),
                "completed_at": final_state.get("completed_at"),
            }

        except Exception as e:
            logger.error(f"TradingOrchestrator failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "symbol": symbol,
            }

    def _count_agents_executed(self, state: TradingState) -> list[str]:
        """Count which agents were executed."""
        executed = []

        agent_keys = [
            ("market_scan", "market_scanner"),
            ("fundamental_analysis", "research_agent"),
            ("technical_analysis", "technical_analyst"),
            ("sentiment_analysis", "sentiment_analyst"),
            ("portfolio_recommendation", "portfolio_manager"),
            ("risk_assessment", "risk_manager"),
            ("trading_signal", "trading_execution"),
        ]

        for state_key, agent_name in agent_keys:
            if state.get(state_key):
                executed.append(agent_name)

        return executed

    async def arun(
        self,
        query: str,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Async version of run.

        Args:
            query: User's query
            symbol: Optional stock symbol

        Returns:
            Analysis results
        """
        # For now, delegate to sync version
        # TODO: Implement true async with parallel agent execution
        return self.run(query, symbol)


# =============================================================================
# Factory Functions
# =============================================================================


def create_trading_orchestrator(
    model_name: str | None = None,
    config: AgentConfig | None = None,
) -> TradingOrchestrator:
    """Create a trading orchestrator instance.

    Args:
        model_name: Optional model name override
        config: Optional configuration

    Returns:
        Configured TradingOrchestrator
    """
    return TradingOrchestrator(config=config, model_name=model_name)


def run_trading_analysis(
    query: str,
    symbol: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Convenience function to run trading analysis.

    Args:
        query: User's analysis request
        symbol: Stock symbol (optional)
        model_name: Model name (optional)

    Returns:
        Analysis results
    """
    orchestrator = create_trading_orchestrator(model_name=model_name)
    return orchestrator.run(query, symbol)
