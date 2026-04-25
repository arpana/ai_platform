import pytest
from ai_platform.tools import BaseTool
from ai_platform.core.models import ToolResult


class ConcreteTool(BaseTool):
    """A concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test_tool"

    @property
    def description(self) -> str:
        return "A test tool"

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {"input": {"type": "string"}}}

    async def execute(self, input_data: dict) -> ToolResult:
        return ToolResult(tool_name=self.name, output="test output")


def test_base_tool_is_abstract():
    """Verify BaseTool cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseTool()


def test_concrete_tool_properties():
    """Verify a concrete tool has all required properties."""
    tool = ConcreteTool()
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert isinstance(tool.input_schema, dict)
    assert tool.environment_scopes == []  # Default is empty


@pytest.mark.asyncio
async def test_concrete_tool_execute():
    """Verify a concrete tool can execute."""
    tool = ConcreteTool()
    result = await tool.execute({"input": "test"})
    assert isinstance(result, ToolResult)
    assert result.tool_name == "test_tool"
    assert result.output == "test output"


def test_custom_environment_scopes():
    """Verify environment_scopes can be overridden."""

    class ScopedTool(BaseTool):
        @property
        def name(self) -> str:
            return "scoped_tool"

        @property
        def description(self) -> str:
            return "A scoped tool"

        @property
        def input_schema(self) -> dict:
            return {}

        @property
        def environment_scopes(self) -> list[str]:
            return ["banking", "retail"]

        async def execute(self, input_data: dict) -> ToolResult:
            return ToolResult(tool_name=self.name, output="scoped")

    tool = ScopedTool()
    assert tool.environment_scopes == ["banking", "retail"]
