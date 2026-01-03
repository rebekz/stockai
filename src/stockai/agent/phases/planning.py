"""Planning Phase for StockAI Agent.

Decomposes complex queries into structured research tasks.
"""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskPlan(BaseModel):
    """Represents a single task in the research plan."""

    id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="What this task accomplishes")
    tools: list[str] = Field(default_factory=list, description="Required tools")
    dependencies: list[str] = Field(default_factory=list, description="Dependent task IDs")
    status: str = Field(default="pending", description="Task status")


class ResearchPlan(BaseModel):
    """Complete research plan for a query."""

    query: str = Field(..., description="Original user query")
    symbol: str | None = Field(default=None, description="Target stock symbol")
    tasks: list[TaskPlan] = Field(default_factory=list)


class PlanningPhase:
    """Handles the planning phase of agent execution.

    Responsibilities:
    - Extract stock symbol from query
    - Decompose query into research tasks
    - Identify required tools for each task
    - Establish task dependencies
    """

    def __init__(self, available_tools: list[str] | None = None):
        """Initialize planning phase.

        Args:
            available_tools: List of available tool names
        """
        self.available_tools = available_tools or []

    def extract_symbol(self, query: str) -> str | None:
        """Extract stock symbol from query.

        Args:
            query: User query text

        Returns:
            Extracted symbol or None
        """
        # Common IDX stocks pattern (4 letters, all caps)
        pattern = r"\b([A-Z]{4})\b"
        matches = re.findall(pattern, query.upper())

        # Known IDX stocks for validation
        known_stocks = {
            "BBCA", "BBRI", "BMRI", "BBNI", "BBTN",
            "TLKM", "ASII", "UNVR", "ICBP", "INDF",
            "GGRM", "HMSP", "KLBF", "PGAS", "SMGR",
            "ADRO", "ANTM", "INCO", "PTBA", "ITMG",
            "ACES", "MAPI", "ERAA", "CPIN", "JPFA",
        }

        for match in matches:
            if match in known_stocks:
                return match

        # Return first 4-letter match if any
        return matches[0] if matches else None

    def create_default_plan(self, query: str, symbol: str | None) -> ResearchPlan:
        """Create default research plan for stock analysis.

        Args:
            query: Original query
            symbol: Target stock symbol

        Returns:
            Default research plan
        """
        tasks = []

        if symbol:
            # Standard stock analysis tasks
            tasks.append(TaskPlan(
                id="task_1",
                description=f"Fetch basic stock information for {symbol}",
                tools=["get_stock_info"],
                dependencies=[],
            ))

            tasks.append(TaskPlan(
                id="task_2",
                description=f"Get current price and recent movement for {symbol}",
                tools=["get_current_price"],
                dependencies=["task_1"],
            ))

            tasks.append(TaskPlan(
                id="task_3",
                description=f"Fetch 1-month price history for {symbol}",
                tools=["get_price_history"],
                dependencies=["task_1"],
            ))

            # Check for technical analysis keywords
            if any(kw in query.lower() for kw in ["technical", "rsi", "macd", "indicator"]):
                tasks.append(TaskPlan(
                    id="task_4",
                    description=f"Calculate technical indicators for {symbol}",
                    tools=["get_technical_indicators"],
                    dependencies=["task_3"],
                ))

            # Check for comparison keywords
            if "compare" in query.lower() or "vs" in query.lower():
                tasks.append(TaskPlan(
                    id="task_5",
                    description="Compare with peer stocks",
                    tools=["compare_stocks"],
                    dependencies=["task_2"],
                ))

        else:
            # Index/general query
            if "idx30" in query.lower():
                tasks.append(TaskPlan(
                    id="task_1",
                    description="Get IDX30 index components",
                    tools=["get_idx30_stocks"],
                    dependencies=[],
                ))
            elif "lq45" in query.lower():
                tasks.append(TaskPlan(
                    id="task_1",
                    description="Get LQ45 index components",
                    tools=["get_lq45_stocks"],
                    dependencies=[],
                ))
            else:
                # Generic fallback
                tasks.append(TaskPlan(
                    id="task_1",
                    description="Get market information",
                    tools=["get_idx30_stocks"],
                    dependencies=[],
                ))

        return ResearchPlan(query=query, symbol=symbol, tasks=tasks)

    def parse_llm_plan(self, llm_response: str, query: str) -> ResearchPlan:
        """Parse plan from LLM response.

        Args:
            llm_response: LLM-generated plan text
            query: Original query

        Returns:
            Parsed research plan
        """
        try:
            # Extract JSON from markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", llm_response)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = llm_response

            task_data = json.loads(json_str)

            # Handle list of tasks
            if isinstance(task_data, list):
                tasks = [TaskPlan(**t) for t in task_data]
            else:
                tasks = [TaskPlan(**t) for t in task_data.get("tasks", [])]

            symbol = self.extract_symbol(query)
            return ResearchPlan(query=query, symbol=symbol, tasks=tasks)

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM plan: {e}")
            # Fallback to default plan
            symbol = self.extract_symbol(query)
            return self.create_default_plan(query, symbol)

    def to_dict(self, plan: ResearchPlan) -> dict[str, Any]:
        """Convert plan to dictionary for state.

        Args:
            plan: Research plan

        Returns:
            Dictionary representation
        """
        return {
            "query": plan.query,
            "symbol": plan.symbol,
            "tasks": [t.model_dump() for t in plan.tasks],
        }
