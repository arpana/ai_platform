from abc import ABC, abstractmethod
from typing import Any

from ai_platform.core.models import ToolResult


class BaseTool(ABC):
    """
    Abstract base class for all tools in the AI Platform.
    
    Each tool must define:
    - name: Unique identifier for the tool
    - description: What the tool does (used by LLM for tool selection)
    - input_schema: JSON Schema describing expected input parameters
    - environment_scopes: Which environments can use this tool (e.g., ["banking"], ["retail"])
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does (for LLM tool selection)."""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema describing the expected input parameters."""
        pass
    
    @property
    def environment_scopes(self) -> list[str]:
        """
        Which environments can use this tool.
        Default: empty list means available to all environments.
        Override to restrict to specific environments like ["banking"] or ["retail"].
        """
        return []
    
    @abstractmethod
    async def execute(self, input_data: dict[str, Any]) -> ToolResult:
        """
        Execute the tool with the given input.
        
        Args:
            input_data: Dictionary containing the tool's input parameters
            
        Returns:
            ToolResult with output or error information
        """
        pass
