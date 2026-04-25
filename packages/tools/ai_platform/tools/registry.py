from ai_platform.core.exceptions import ToolExecutionError, ToolNotFoundError
from ai_platform.core.models import ToolDefinition

from .base import BaseTool


class ToolRegistry:
    """
    Central registry for managing tool registration, lookup, and environment-scoped filtering.

    This registry is the source of truth for all available tools in the platform.
    Tools are registered programmatically at startup and filtered by environment scope.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: The tool instance to register

        Raises:
            ToolExecutionError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ToolExecutionError(
                f"Tool '{tool.name}' is already registered. Cannot overwrite.",
                code="DUPLICATE_TOOL",
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """
        Retrieve a tool by name.

        Args:
            name: The tool name to look up

        Returns:
            The tool instance

        Raises:
            ToolNotFoundError: If the tool is not registered
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found in registry", code="TOOL_NOT_FOUND")
        return self._tools[name]

    def list_tools(self) -> list[BaseTool]:
        """
        Get all registered tools.

        Returns:
            List of all tool instances
        """
        return list(self._tools.values())

    def list_for_environment(self, env: str) -> list[BaseTool]:
        """
        Get tools available for a specific environment.

        Args:
            env: Environment name (e.g., "banking", "retail")

        Returns:
            List of tools that are scoped to the given environment.
            Tools with empty environment_scopes are available to all environments.
        """
        return [
            tool
            for tool in self._tools.values()
            if not tool.environment_scopes or env in tool.environment_scopes
        ]

    def get_definitions(self, env: str) -> list[ToolDefinition]:
        """
        Get ToolDefinition objects for all tools in an environment.

        Args:
            env: Environment name (e.g., "banking", "retail")

        Returns:
            List of ToolDefinition models for serialization/API responses
        """
        tools = self.list_for_environment(env)
        return [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
                environment_scopes=tool.environment_scopes,
            )
            for tool in tools
        ]
