"""Agent State Management.

Defines the state schema for the multi-phase agent workflow.
Uses Pydantic for validation and TypedDict for LangGraph compatibility.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Task(BaseModel):
    """Represents a single research task in the agent's plan."""

    id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="What this task accomplishes")
    tools: list[str] = Field(default_factory=list, description="Required tools")
    dependencies: list[str] = Field(default_factory=list, description="Task IDs that must complete first")
    status: str = Field(default="pending", description="pending|running|completed|failed")
    result: Any = Field(default=None, description="Task execution result")
    error: str | None = Field(default=None, description="Error message if failed")
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)


class AgentPlan(BaseModel):
    """The agent's research plan."""

    query: str = Field(..., description="Original user query")
    tasks: list[Task] = Field(default_factory=list, description="Planned tasks")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ValidationResult(BaseModel):
    """Result of the validation phase."""

    status: str = Field(..., description="PASS|RETRY|FAIL")
    issues: list[str] = Field(default_factory=list)
    retry_tasks: list[str] = Field(default_factory=list)
    message: str = Field(default="")


class AgentState(TypedDict, total=False):
    """State schema for the LangGraph workflow.

    This TypedDict defines all state that flows through the agent phases.
    """

    # Input
    query: str
    symbol: str | None

    # Planning phase
    plan: dict | None
    current_task_index: int

    # Action phase
    tool_results: list[dict]
    tool_calls: int

    # Validation phase
    validation: dict | None
    validation_attempts: int

    # Answer phase
    answer: str | None

    # Metadata
    messages: list[dict]
    phase: str
    error: str | None
    started_at: str
    completed_at: str | None


def create_initial_state(query: str, symbol: str | None = None) -> AgentState:
    """Create initial agent state for a new query.

    Args:
        query: User's analysis request
        symbol: Optional stock symbol extracted from query

    Returns:
        Initial AgentState dictionary
    """
    return AgentState(
        query=query,
        symbol=symbol,
        plan=None,
        current_task_index=0,
        tool_results=[],
        tool_calls=0,
        validation=None,
        validation_attempts=0,
        answer=None,
        messages=[],
        phase="planning",
        error=None,
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
    )


class SessionManager:
    """Manages agent session persistence.

    Saves and loads agent state to/from JSON files for conversation continuity.
    """

    def __init__(self, session_dir: Path | None = None):
        """Initialize session manager.

        Args:
            session_dir: Directory for session files (default: ~/.stockai/)
        """
        if session_dir is None:
            session_dir = Path.home() / ".stockai"
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.session_dir / "session.json"

    def save_state(self, state: AgentState) -> bool:
        """Save current agent state to session file.

        Args:
            state: Current agent state

        Returns:
            True if saved successfully
        """
        try:
            # Convert state to JSON-serializable format
            state_dict = dict(state)
            state_dict["saved_at"] = datetime.utcnow().isoformat()

            with open(self.session_file, "w") as f:
                json.dump(state_dict, f, indent=2, default=str)

            logger.debug(f"Session saved to {self.session_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def load_state(self) -> AgentState | None:
        """Load agent state from session file.

        Returns:
            Loaded AgentState or None if no session exists
        """
        if not self.session_file.exists():
            return None

        try:
            with open(self.session_file) as f:
                state_dict = json.load(f)

            # Remove metadata fields not in AgentState
            state_dict.pop("saved_at", None)

            logger.debug(f"Session loaded from {self.session_file}")
            return AgentState(**state_dict)

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def clear_session(self) -> bool:
        """Clear the current session.

        Returns:
            True if cleared successfully
        """
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            logger.debug("Session cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return False

    def has_session(self) -> bool:
        """Check if an active session exists."""
        return self.session_file.exists()
