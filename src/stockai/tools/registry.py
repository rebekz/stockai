"""Tool Registry for StockAI Agent.

Provides a decorator-based system for registering tools that the agent can use.
Tools are automatically discovered and made available to the LLM.
"""

import logging
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class ToolRegistry:
    """Registry for agent tools.

    Manages tool registration, discovery, and access control.

    Attributes:
        tools: Dictionary of registered tools
        permissions: Tool permission levels
    """

    _instance: "ToolRegistry | None" = None

    def __new__(cls) -> "ToolRegistry":
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return
        self.tools: dict[str, dict[str, Any]] = {}
        self.permissions: dict[str, str] = {}
        self._initialized = True
        logger.debug("Tool registry initialized")

    def register(
        self,
        name: str,
        func: Callable,
        description: str | None = None,
        permission: str = "safe",
        category: str = "general",
    ) -> None:
        """Register a tool function.

        Args:
            name: Unique tool name
            func: The tool function
            description: Human-readable description (defaults to docstring)
            permission: Permission level (safe, elevated, dangerous)
            category: Tool category for organization
        """
        if name in self.tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")

        doc = description or func.__doc__ or "No description available"
        # Get first line of docstring
        doc_first_line = doc.split("\n")[0].strip()

        self.tools[name] = {
            "name": name,
            "func": func,
            "description": doc_first_line,
            "full_description": doc,
            "permission": permission,
            "category": category,
        }
        self.permissions[name] = permission

        logger.debug(f"Registered tool: {name} ({category}, {permission})")

    def get_tool(self, name: str) -> Callable | None:
        """Get a tool function by name.

        Args:
            name: Tool name

        Returns:
            Tool function or None if not found
        """
        tool_info = self.tools.get(name)
        if tool_info:
            return tool_info["func"]
        return None

    def get_tool_info(self, name: str) -> dict | None:
        """Get full tool information.

        Args:
            name: Tool name

        Returns:
            Tool info dictionary or None
        """
        return self.tools.get(name)

    def list_tools(self, category: str | None = None) -> list[dict]:
        """List all registered tools.

        Args:
            category: Optional category filter

        Returns:
            List of tool info dictionaries
        """
        tools = list(self.tools.values())
        if category:
            tools = [t for t in tools if t["category"] == category]
        return tools

    def get_tools_dict(self) -> dict[str, Callable]:
        """Get dictionary mapping tool names to functions.

        Returns:
            Dictionary for agent initialization
        """
        return {name: info["func"] for name, info in self.tools.items()}

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for LLM.

        Returns:
            Markdown-formatted tool list
        """
        lines = []
        categories = {}

        # Group by category
        for tool in self.tools.values():
            cat = tool["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool)

        # Format each category
        for cat, tools in sorted(categories.items()):
            lines.append(f"\n### {cat.title()}")
            for tool in tools:
                perm_icon = {"safe": "✅", "elevated": "⚠️", "dangerous": "🚫"}.get(
                    tool["permission"], ""
                )
                lines.append(f"- **{tool['name']}** {perm_icon}: {tool['description']}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all registered tools."""
        self.tools.clear()
        self.permissions.clear()
        logger.debug("Tool registry cleared")


def stockai_tool(
    name: str | None = None,
    description: str | None = None,
    permission: str = "safe",
    category: str = "general",
) -> Callable[[F], F]:
    """Decorator to register a function as an agent tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        permission: Permission level (safe, elevated, dangerous)
        category: Tool category

    Returns:
        Decorated function

    Example:
        @stockai_tool(category="data")
        def get_stock_info(symbol: str) -> dict:
            '''Get basic stock information.'''
            ...
    """
    def decorator(func: F) -> F:
        tool_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Tool called: {tool_name}")
            return func(*args, **kwargs)

        # Register with global registry
        registry = get_registry()
        registry.register(
            name=tool_name,
            func=wrapper,
            description=description,
            permission=permission,
            category=category,
        )

        return wrapper

    return decorator


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance.

    Returns:
        Singleton ToolRegistry
    """
    return ToolRegistry()


def get_all_tools() -> dict[str, Callable]:
    """Get all registered tools as a dictionary.

    Returns:
        Dictionary mapping tool names to functions
    """
    return get_registry().get_tools_dict()
