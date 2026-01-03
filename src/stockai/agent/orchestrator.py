"""StockAI Agent Orchestrator.

Main agent class that coordinates the multi-phase research workflow:
Planning → Action → Validation → Answer

Uses LangChain with Google Gemini for LLM capabilities
and LangGraph for workflow orchestration.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from stockai.agent.prompts import (
    SYSTEM_PROMPT,
    PLANNING_PROMPT,
    ACTION_PROMPT,
    VALIDATION_PROMPT,
    ANSWER_PROMPT,
)
from stockai.agent.state import AgentState, SessionManager, create_initial_state
from stockai.config import get_settings

logger = logging.getLogger(__name__)


class StockAIAgent:
    """Autonomous research agent for Indonesian stock analysis.

    This agent implements a multi-phase workflow:
    1. Planning: Decompose query into research tasks
    2. Action: Execute tools to gather data
    3. Validation: Verify data completeness
    4. Answer: Synthesize findings into response

    Attributes:
        model: LangChain LLM instance
        tools: Registered tool functions
        session: Session persistence manager
        max_tool_calls: Safety limit for tool executions
        max_validation_attempts: Max retry loops
    """

    def __init__(
        self,
        model_name: str | None = None,
        tools: dict[str, Callable] | None = None,
        session_manager: SessionManager | None = None,
    ):
        """Initialize the StockAI agent.

        Args:
            model_name: LLM model to use (default from settings)
            tools: Dictionary of tool functions
            session_manager: Session persistence manager
        """
        settings = get_settings()
        self.model_name = model_name or settings.model

        # Initialize LLM
        self.llm = self._create_llm()

        # Tool registry
        self.tools = tools or {}

        # Session management
        self.session = session_manager or SessionManager()

        # Safety limits
        self.max_tool_calls = 10
        self.max_validation_attempts = 2

        # Build workflow graph
        self.workflow = self._build_workflow()

        logger.info(f"StockAI Agent initialized with model: {self.model_name}")

    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Create the LangChain LLM instance.

        Returns:
            Configured ChatGoogleGenerativeAI instance
        """
        settings = get_settings()

        # Map model names to Gemini model IDs
        model_mapping = {
            "gemini-3-flash-preview": "gemini-2.0-flash",
            "gemini-flash": "gemini-2.0-flash",
            "gemini-pro": "gemini-pro",
        }

        model_id = model_mapping.get(self.model_name, "gemini-2.0-flash")

        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=settings.google_api_key,
            temperature=0.3,
            max_tokens=4096,
            convert_system_message_to_human=True,
        )

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            Compiled StateGraph for agent execution
        """
        # Create state graph
        workflow = StateGraph(AgentState)

        # Add nodes for each phase
        workflow.add_node("planning", self._planning_phase)
        workflow.add_node("action", self._action_phase)
        workflow.add_node("validation", self._validation_phase)
        workflow.add_node("answer", self._answer_phase)

        # Define edges
        workflow.set_entry_point("planning")
        workflow.add_edge("planning", "action")
        workflow.add_conditional_edges(
            "action",
            self._should_continue_action,
            {
                "continue": "action",
                "validate": "validation",
            },
        )
        workflow.add_conditional_edges(
            "validation",
            self._validation_router,
            {
                "pass": "answer",
                "retry": "action",
                "fail": "answer",
            },
        )
        workflow.add_edge("answer", END)

        return workflow.compile()

    def _planning_phase(self, state: AgentState) -> dict:
        """Execute the planning phase.

        Decomposes the user query into structured research tasks.

        Args:
            state: Current agent state

        Returns:
            State updates with plan
        """
        logger.info("Planning phase started")

        # Build planning prompt
        tool_descriptions = self._get_tool_descriptions()
        prompt = PLANNING_PROMPT.format(
            query=state["query"],
            tools=tool_descriptions,
            symbol=state.get("symbol", "UNKNOWN"),
        )

        # Get plan from LLM
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        plan_text = response.content

        # Parse plan JSON
        plan = self._parse_plan(plan_text)

        logger.info(f"Planning complete: {len(plan.get('tasks', []))} tasks")

        return {
            "plan": plan,
            "phase": "action",
            "messages": state.get("messages", []) + [
                {"role": "planning", "content": plan_text}
            ],
        }

    def _action_phase(self, state: AgentState) -> dict:
        """Execute the action phase.

        Runs tools based on the current task in the plan.

        Args:
            state: Current agent state

        Returns:
            State updates with tool results
        """
        logger.info("Action phase started")

        plan = state.get("plan", {})
        tasks = plan.get("tasks", [])
        current_idx = state.get("current_task_index", 0)

        if current_idx >= len(tasks):
            # All tasks complete
            return {"phase": "validation"}

        task = tasks[current_idx]
        logger.info(f"Executing task {task.get('id')}: {task.get('description')}")

        # Execute tools for this task
        tool_results = state.get("tool_results", [])
        tool_calls = state.get("tool_calls", 0)

        for tool_name in task.get("tools", []):
            if tool_calls >= self.max_tool_calls:
                logger.warning("Max tool calls reached")
                break

            result = self._execute_tool(tool_name, state)
            tool_results.append({
                "task_id": task.get("id"),
                "tool": tool_name,
                "result": result,
                "timestamp": datetime.utcnow().isoformat(),
            })
            tool_calls += 1

        # Mark task complete and move to next
        task["status"] = "completed"
        tasks[current_idx] = task

        return {
            "plan": {**plan, "tasks": tasks},
            "current_task_index": current_idx + 1,
            "tool_results": tool_results,
            "tool_calls": tool_calls,
            "phase": "action",
        }

    def _validation_phase(self, state: AgentState) -> dict:
        """Execute the validation phase.

        Verifies data completeness and quality.

        Args:
            state: Current agent state

        Returns:
            State updates with validation result
        """
        logger.info("Validation phase started")

        # Build validation prompt
        prompt = VALIDATION_PROMPT.format(
            completed_tasks=json.dumps(state.get("plan", {}).get("tasks", []), default=str),
            data=json.dumps(state.get("tool_results", []), default=str),
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        validation_text = response.content

        # Parse validation result
        validation = self._parse_validation(validation_text)

        attempts = state.get("validation_attempts", 0) + 1
        logger.info(f"Validation: {validation.get('status')} (attempt {attempts})")

        return {
            "validation": validation,
            "validation_attempts": attempts,
            "phase": "validation",
        }

    def _answer_phase(self, state: AgentState) -> dict:
        """Execute the answer phase.

        Synthesizes collected data into final response.

        Args:
            state: Current agent state

        Returns:
            State updates with answer
        """
        logger.info("Answer phase started")

        # Build answer prompt
        prompt = ANSWER_PROMPT.format(
            query=state["query"],
            data=json.dumps(state.get("tool_results", []), default=str, indent=2),
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        answer = response.content

        logger.info("Answer generation complete")

        return {
            "answer": answer,
            "phase": "complete",
            "completed_at": datetime.utcnow().isoformat(),
        }

    def _should_continue_action(self, state: AgentState) -> str:
        """Determine if action phase should continue.

        Args:
            state: Current agent state

        Returns:
            "continue" if more tasks, "validate" if done
        """
        plan = state.get("plan", {})
        tasks = plan.get("tasks", [])
        current_idx = state.get("current_task_index", 0)

        if current_idx >= len(tasks):
            return "validate"
        return "continue"

    def _validation_router(self, state: AgentState) -> str:
        """Route based on validation result.

        Args:
            state: Current agent state

        Returns:
            "pass", "retry", or "fail"
        """
        validation = state.get("validation", {})
        status = validation.get("status", "FAIL").upper()
        attempts = state.get("validation_attempts", 0)

        if status == "PASS":
            return "pass"
        elif status == "RETRY" and attempts < self.max_validation_attempts:
            return "retry"
        else:
            return "fail"

    def _execute_tool(self, tool_name: str, state: AgentState) -> Any:
        """Execute a registered tool.

        Args:
            tool_name: Name of tool to execute
            state: Current agent state for context

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            logger.warning(f"Tool not found: {tool_name}")
            return {"error": f"Tool '{tool_name}' not registered"}

        try:
            tool_func = self.tools[tool_name]
            symbol = state.get("symbol")

            # Call tool with symbol if available
            if symbol:
                result = tool_func(symbol)
            else:
                result = tool_func()

            return result

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return {"error": str(e)}

    def _get_tool_descriptions(self) -> str:
        """Get formatted descriptions of available tools.

        Returns:
            Markdown-formatted tool list
        """
        if not self.tools:
            return "No tools registered."

        descriptions = []
        for name, func in self.tools.items():
            doc = func.__doc__ or "No description"
            descriptions.append(f"- **{name}**: {doc.split(chr(10))[0]}")

        return "\n".join(descriptions)

    def _parse_plan(self, text: str) -> dict:
        """Parse plan JSON from LLM response.

        Args:
            text: LLM response text

        Returns:
            Parsed plan dictionary
        """
        try:
            # Extract JSON from markdown code blocks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Try to find JSON array directly
                json_str = text

            tasks = json.loads(json_str)

            if isinstance(tasks, list):
                return {"tasks": tasks}
            return tasks

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan: {e}")
            # Create minimal fallback plan
            return {
                "tasks": [
                    {
                        "id": "task_1",
                        "description": "Gather basic stock information",
                        "tools": ["get_stock_info"],
                        "dependencies": [],
                        "status": "pending",
                    }
                ]
            }

    def _parse_validation(self, text: str) -> dict:
        """Parse validation result from LLM response.

        Args:
            text: LLM response text

        Returns:
            Parsed validation dictionary
        """
        try:
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if json_match:
                json_str = json_match.group(1).strip()
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Fallback: parse status from text
        if "PASS" in text.upper():
            return {"status": "PASS", "issues": [], "message": text}
        elif "RETRY" in text.upper():
            return {"status": "RETRY", "issues": [], "retry_tasks": [], "message": text}
        else:
            return {"status": "FAIL", "issues": [text], "message": text}

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool function.

        Args:
            name: Tool name for LLM reference
            func: Callable tool function
        """
        self.tools[name] = func
        logger.debug(f"Registered tool: {name}")

    def run(self, query: str, symbol: str | None = None) -> dict:
        """Run the agent on a query.

        Args:
            query: User's analysis request
            symbol: Optional stock symbol

        Returns:
            Agent result with answer and metadata
        """
        logger.info(f"Agent starting: {query}")

        # Create initial state
        state = create_initial_state(query, symbol)

        # Run workflow
        try:
            final_state = self.workflow.invoke(state)

            # Save session
            self.session.save_state(final_state)

            return {
                "success": True,
                "answer": final_state.get("answer"),
                "plan": final_state.get("plan"),
                "tool_results": final_state.get("tool_results"),
                "phases_completed": final_state.get("phase"),
                "duration": self._calculate_duration(
                    final_state.get("started_at"),
                    final_state.get("completed_at"),
                ),
            }

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": None,
            }

    def _calculate_duration(self, start: str | None, end: str | None) -> float | None:
        """Calculate execution duration in seconds."""
        if not start or not end:
            return None
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            return (end_dt - start_dt).total_seconds()
        except (ValueError, TypeError):
            return None


def create_agent(
    model_name: str | None = None,
    tools: dict[str, Callable] | None = None,
) -> StockAIAgent:
    """Factory function to create a StockAI agent.

    Args:
        model_name: LLM model to use
        tools: Dictionary of tool functions

    Returns:
        Configured StockAIAgent instance
    """
    return StockAIAgent(model_name=model_name, tools=tools)
