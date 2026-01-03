"""Action Phase for StockAI Agent.

Executes tools based on the research plan.
"""

import logging
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ActionPhase:
    """Handles the action phase of agent execution.

    Responsibilities:
    - Execute tools for each task
    - Handle tool errors with retry logic
    - Collect and store results
    - Respect safety limits
    """

    def __init__(
        self,
        tools: dict[str, Callable],
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ):
        """Initialize action phase.

        Args:
            tools: Dictionary of available tools
            max_retries: Maximum retry attempts per tool
            timeout_seconds: Timeout per tool call
        """
        self.tools = tools
        self.max_retries = max_retries
        self.timeout = timeout_seconds

    def execute_task(
        self,
        task: dict,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single task from the plan.

        Args:
            task: Task dictionary with id, tools, etc.
            context: Current execution context (symbol, prior results)

        Returns:
            Task execution result
        """
        task_id = task.get("id", "unknown")
        logger.info(f"Executing task: {task_id}")

        results = []
        errors = []

        for tool_name in task.get("tools", []):
            result = self._execute_tool_with_retry(tool_name, context)
            if "error" in result:
                errors.append(result)
            else:
                results.append(result)

        return {
            "task_id": task_id,
            "tools_executed": task.get("tools", []),
            "results": results,
            "errors": errors,
            "success": len(errors) == 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _execute_tool_with_retry(
        self,
        tool_name: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool with retry logic.

        Args:
            tool_name: Name of tool to execute
            context: Execution context

        Returns:
            Tool result or error
        """
        if tool_name not in self.tools:
            logger.warning(f"Tool not found: {tool_name}")
            return {"error": f"Tool '{tool_name}' not available", "tool": tool_name}

        tool_func = self.tools[tool_name]
        symbol = context.get("symbol")

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Tool {tool_name} attempt {attempt + 1}")

                # Execute tool with appropriate arguments
                if symbol:
                    result = tool_func(symbol)
                else:
                    result = tool_func()

                # Validate result
                if result is None:
                    raise ValueError("Tool returned None")

                return {
                    "tool": tool_name,
                    "data": result,
                    "attempt": attempt + 1,
                }

            except Exception as e:
                logger.warning(f"Tool {tool_name} failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return {
                        "error": str(e),
                        "tool": tool_name,
                        "attempts": self.max_retries,
                    }

        return {"error": "Max retries exceeded", "tool": tool_name}

    def get_ready_tasks(
        self,
        tasks: list[dict],
        completed_task_ids: set[str],
    ) -> list[dict]:
        """Get tasks that are ready to execute (dependencies met).

        Args:
            tasks: All tasks in plan
            completed_task_ids: Set of completed task IDs

        Returns:
            List of ready tasks
        """
        ready = []

        for task in tasks:
            if task.get("status") == "completed":
                continue

            dependencies = set(task.get("dependencies", []))
            if dependencies.issubset(completed_task_ids):
                ready.append(task)

        return ready

    def format_results_for_context(
        self,
        tool_results: list[dict],
    ) -> str:
        """Format tool results for LLM context.

        Args:
            tool_results: List of tool execution results

        Returns:
            Formatted string for LLM
        """
        lines = []

        for result in tool_results:
            task_id = result.get("task_id", "unknown")
            lines.append(f"\n## Task: {task_id}")

            for tool_result in result.get("results", []):
                tool = tool_result.get("tool", "unknown")
                data = tool_result.get("data", {})
                lines.append(f"\n### {tool}")
                lines.append(f"```json\n{data}\n```")

            for error in result.get("errors", []):
                lines.append(f"\n### Error: {error.get('tool')}")
                lines.append(f"Error: {error.get('error')}")

        return "\n".join(lines)
