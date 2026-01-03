"""Conversation Memory for StockAI Agent.

Manages conversation history and context for multi-turn interactions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Represents a single message in conversation."""

    role: str = Field(..., description="user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMemory:
    """Manages conversation history for the agent.

    Provides:
    - Message storage and retrieval
    - Context window management
    - Conversation persistence
    """

    def __init__(
        self,
        max_messages: int = 50,
        storage_path: Path | None = None,
    ):
        """Initialize conversation memory.

        Args:
            max_messages: Maximum messages to retain
            storage_path: Path for conversation persistence
        """
        self.max_messages = max_messages
        self.messages: list[Message] = []

        if storage_path is None:
            storage_path = Path.home() / ".stockai" / "conversations"
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.conversation_id: str | None = None

    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to conversation history.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(message)

        # Trim if needed
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        logger.debug(f"Added message: {role} ({len(content)} chars)")

    def get_messages(
        self,
        last_n: int | None = None,
        role_filter: str | None = None,
    ) -> list[dict]:
        """Get conversation messages.

        Args:
            last_n: Get last N messages (default all)
            role_filter: Filter by role

        Returns:
            List of message dictionaries
        """
        messages = self.messages

        if role_filter:
            messages = [m for m in messages if m.role == role_filter]

        if last_n:
            messages = messages[-last_n:]

        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in messages
        ]

    def get_context_window(self, max_tokens: int = 4000) -> list[dict]:
        """Get messages that fit within token budget.

        Args:
            max_tokens: Approximate token limit

        Returns:
            Messages within budget
        """
        # Rough estimate: 4 chars per token
        char_budget = max_tokens * 4
        messages = []
        total_chars = 0

        # Start from most recent
        for message in reversed(self.messages):
            msg_chars = len(message.content)
            if total_chars + msg_chars > char_budget:
                break
            messages.insert(0, {
                "role": message.role,
                "content": message.content,
            })
            total_chars += msg_chars

        return messages

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
        logger.debug("Conversation memory cleared")

    def save(self, conversation_id: str | None = None) -> bool:
        """Save conversation to file.

        Args:
            conversation_id: Unique ID for conversation

        Returns:
            True if saved successfully
        """
        if conversation_id is None:
            conversation_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        self.conversation_id = conversation_id
        file_path = self.storage_path / f"{conversation_id}.json"

        try:
            data = {
                "id": conversation_id,
                "messages": [m.model_dump() for m in self.messages],
                "saved_at": datetime.utcnow().isoformat(),
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Conversation saved: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False

    def load(self, conversation_id: str) -> bool:
        """Load conversation from file.

        Args:
            conversation_id: Conversation ID to load

        Returns:
            True if loaded successfully
        """
        file_path = self.storage_path / f"{conversation_id}.json"

        if not file_path.exists():
            logger.warning(f"Conversation not found: {conversation_id}")
            return False

        try:
            with open(file_path) as f:
                data = json.load(f)

            self.messages = [
                Message(**m) for m in data.get("messages", [])
            ]
            self.conversation_id = conversation_id

            logger.debug(f"Loaded {len(self.messages)} messages")
            return True

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False

    def list_conversations(self) -> list[dict]:
        """List all saved conversations.

        Returns:
            List of conversation metadata
        """
        conversations = []

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    conversations.append({
                        "id": data.get("id"),
                        "messages": len(data.get("messages", [])),
                        "saved_at": data.get("saved_at"),
                    })
            except Exception:
                continue

        return sorted(conversations, key=lambda x: x.get("saved_at", ""), reverse=True)

    def get_summary(self) -> str:
        """Get a summary of current conversation.

        Returns:
            Summary string
        """
        if not self.messages:
            return "No conversation history."

        user_msgs = len([m for m in self.messages if m.role == "user"])
        assistant_msgs = len([m for m in self.messages if m.role == "assistant"])

        first_msg = self.messages[0].timestamp
        last_msg = self.messages[-1].timestamp

        return (
            f"Conversation with {user_msgs} user messages and {assistant_msgs} responses. "
            f"Started: {first_msg.strftime('%H:%M')}, Last: {last_msg.strftime('%H:%M')}"
        )
