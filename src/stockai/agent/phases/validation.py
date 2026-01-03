"""Validation Phase for StockAI Agent.

Verifies data completeness and quality before answer synthesis.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of the validation phase."""

    status: str = Field(..., description="PASS, RETRY, or FAIL")
    issues: list[str] = Field(default_factory=list)
    retry_tasks: list[str] = Field(default_factory=list)
    message: str = Field(default="")
    data_quality_score: float = Field(default=0.0)


class ValidationPhase:
    """Handles the validation phase of agent execution.

    Responsibilities:
    - Verify all planned tasks completed
    - Check data quality (no empty/null results)
    - Identify missing or invalid data
    - Request re-execution if needed
    """

    def __init__(self, max_attempts: int = 2):
        """Initialize validation phase.

        Args:
            max_attempts: Maximum validation/retry cycles
        """
        self.max_attempts = max_attempts

    def validate(
        self,
        tasks: list[dict],
        tool_results: list[dict],
    ) -> ValidationResult:
        """Validate the collected research data.

        Args:
            tasks: Planned tasks
            tool_results: Collected tool results

        Returns:
            Validation result
        """
        issues = []
        retry_tasks = []
        quality_score = 1.0

        # Check task completion
        completed_tasks = {r.get("task_id") for r in tool_results}
        all_task_ids = {t.get("id") for t in tasks}

        missing_tasks = all_task_ids - completed_tasks
        if missing_tasks:
            issues.append(f"Missing tasks: {missing_tasks}")
            retry_tasks.extend(missing_tasks)
            quality_score -= 0.2 * len(missing_tasks)

        # Check for errors
        error_count = 0
        for result in tool_results:
            errors = result.get("errors", [])
            if errors:
                error_count += len(errors)
                task_id = result.get("task_id")
                issues.append(f"Errors in {task_id}: {[e.get('error') for e in errors]}")
                if task_id not in retry_tasks:
                    retry_tasks.append(task_id)

        if error_count > 0:
            quality_score -= 0.1 * error_count

        # Check data quality
        for result in tool_results:
            for tool_result in result.get("results", []):
                data = tool_result.get("data", {})

                # Check for empty data
                if not data or (isinstance(data, dict) and "error" in data):
                    task_id = result.get("task_id")
                    issues.append(f"Empty or error data in {task_id}")
                    quality_score -= 0.15
                    if task_id not in retry_tasks:
                        retry_tasks.append(task_id)

        # Determine status
        quality_score = max(0.0, quality_score)

        if len(issues) == 0:
            status = "PASS"
            message = "All data validated successfully"
        elif len(retry_tasks) > 0 and quality_score > 0.3:
            status = "RETRY"
            message = f"Some issues found, retry recommended: {retry_tasks}"
        else:
            # Low quality or too many issues
            status = "FAIL" if quality_score < 0.3 else "PASS"
            message = f"Validation complete with issues: {issues}"

        logger.info(f"Validation: {status} (score: {quality_score:.2f})")

        return ValidationResult(
            status=status,
            issues=issues,
            retry_tasks=retry_tasks,
            message=message,
            data_quality_score=quality_score,
        )

    def check_required_data(
        self,
        tool_results: list[dict],
        required_fields: list[str],
    ) -> list[str]:
        """Check if required data fields are present.

        Args:
            tool_results: Collected results
            required_fields: List of required field names

        Returns:
            List of missing fields
        """
        # Flatten all data
        all_data = {}
        for result in tool_results:
            for tool_result in result.get("results", []):
                data = tool_result.get("data", {})
                if isinstance(data, dict):
                    all_data.update(data)

        missing = []
        for field in required_fields:
            if field not in all_data or all_data[field] is None:
                missing.append(field)

        return missing

    def summarize_data(self, tool_results: list[dict]) -> dict[str, Any]:
        """Create a summary of collected data.

        Args:
            tool_results: Collected results

        Returns:
            Summary dictionary
        """
        summary = {
            "total_tasks": len(tool_results),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_data_points": 0,
            "tools_used": set(),
        }

        for result in tool_results:
            if result.get("success", False):
                summary["successful_tasks"] += 1
            else:
                summary["failed_tasks"] += 1

            for tool_result in result.get("results", []):
                summary["tools_used"].add(tool_result.get("tool"))
                data = tool_result.get("data", {})
                if isinstance(data, dict):
                    summary["total_data_points"] += len(data)

        summary["tools_used"] = list(summary["tools_used"])

        return summary
