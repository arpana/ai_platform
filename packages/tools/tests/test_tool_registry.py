import pytest
from ai_platform.tools import BaseTool, ToolRegistry
from ai_platform.core.models import ToolResult
from ai_platform.core.exceptions import ToolExecutionError, ToolNotFoundError


class MockBankingTool(BaseTool):
    @property
    def name(self) -> str:
        return "banking_tool"
    
    @property
    def description(self) -> str:
        return "Banking tool"
    
    @property
    def input_schema(self) -> dict:
        return {}
    
    @property
    def environment_scopes(self) -> list[str]:
        return ["banking"]
    
    async def execute(self, input_data: dict) -> ToolResult:
        return ToolResult(tool_name=self.name, output="banking result")


class MockRetailTool(BaseTool):
    @property
    def name(self) -> str:
        return "retail_tool"
    
    @property
    def description(self) -> str:
        return "Retail tool"
    
    @property
    def input_schema(self) -> dict:
        return {}
    
    @property
    def environment_scopes(self) -> list[str]:
        return ["retail"]
    
    async def execute(self, input_data: dict) -> ToolResult:
        return ToolResult(tool_name=self.name, output="retail result")


class MockUniversalTool(BaseTool):
    @property
    def name(self) -> str:
        return "universal_tool"
    
    @property
    def description(self) -> str:
        return "Universal tool"
    
    @property
    def input_schema(self) -> dict:
        return {}
    
    @property
    def environment_scopes(self) -> list[str]:
        return []  # Available to all environments
    
    async def execute(self, input_data: dict) -> ToolResult:
        return ToolResult(tool_name=self.name, output="universal result")


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return ToolRegistry()


@pytest.fixture
def populated_registry():
    """Create a registry with tools already registered."""
    registry = ToolRegistry()
    registry.register(MockBankingTool())
    registry.register(MockRetailTool())
    registry.register(MockUniversalTool())
    return registry


def test_register_tool(registry):
    """Test registering a tool."""
    tool = MockBankingTool()
    registry.register(tool)
    assert registry.get("banking_tool") == tool


def test_register_duplicate_tool_raises_error(registry):
    """Test that registering a duplicate tool raises ToolExecutionError."""
    registry.register(MockBankingTool())
    with pytest.raises(ToolExecutionError) as exc_info:
        registry.register(MockBankingTool())
    assert "already registered" in str(exc_info.value)


def test_get_tool(populated_registry):
    """Test retrieving a tool by name."""
    tool = populated_registry.get("banking_tool")
    assert tool.name == "banking_tool"


def test_get_nonexistent_tool_raises_error(registry):
    """Test that getting a non-existent tool raises ToolNotFoundError."""
    with pytest.raises(ToolNotFoundError) as exc_info:
        registry.get("nonexistent_tool")
    assert "not found" in str(exc_info.value)


def test_list_tools(populated_registry):
    """Test listing all registered tools."""
    tools = populated_registry.list_tools()
    assert len(tools) == 3
    tool_names = {t.name for t in tools}
    assert tool_names == {"banking_tool", "retail_tool", "universal_tool"}


def test_list_for_environment_banking(populated_registry):
    """Test listing tools for banking environment."""
    tools = populated_registry.list_for_environment("banking")
    tool_names = {t.name for t in tools}
    assert "banking_tool" in tool_names
    assert "universal_tool" in tool_names
    assert "retail_tool" not in tool_names


def test_list_for_environment_retail(populated_registry):
    """Test listing tools for retail environment."""
    tools = populated_registry.list_for_environment("retail")
    tool_names = {t.name for t in tools}
    assert "retail_tool" in tool_names
    assert "universal_tool" in tool_names
    assert "banking_tool" not in tool_names


def test_list_for_environment_empty(registry):
    """Test listing tools for an environment with no registered tools."""
    tools = registry.list_for_environment("banking")
    assert len(tools) == 0


def test_get_definitions(populated_registry):
    """Test getting ToolDefinition objects for an environment."""
    definitions = populated_registry.get_definitions("banking")
    assert len(definitions) == 2
    def_names = {d.name for d in definitions}
    assert def_names == {"banking_tool", "universal_tool"}
    
    # Verify it's the correct model type
    for definition in definitions:
        assert hasattr(definition, 'name')
        assert hasattr(definition, 'description')
        assert hasattr(definition, 'input_schema')
        assert hasattr(definition, 'environment_scopes')


def test_list_for_environment_with_multiple_scopes(registry):
    """Test a tool with multiple environment scopes."""
    
    class MultiScopeTool(BaseTool):
        @property
        def name(self) -> str:
            return "multi_scope"
        
        @property
        def description(self) -> str:
            return "Multi-scope tool"
        
        @property
        def input_schema(self) -> dict:
            return {}
        
        @property
        def environment_scopes(self) -> list[str]:
            return ["banking", "retail"]
        
        async def execute(self, input_data: dict) -> ToolResult:
            return ToolResult(tool_name=self.name, output="multi")
    
    registry.register(MultiScopeTool())
    
    banking_tools = registry.list_for_environment("banking")
    retail_tools = registry.list_for_environment("retail")
    
    assert any(t.name == "multi_scope" for t in banking_tools)
    assert any(t.name == "multi_scope" for t in retail_tools)
