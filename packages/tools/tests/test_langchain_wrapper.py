import pytest
from ai_platform.tools import (
    LoanCheckerTool,
    OrderStatusTool,
    RecommendationEngineTool,
    ToolRegistry,
    wrap_as_langchain_tool,
    wrap_tools_for_environment
)
from langchain_core.tools import StructuredTool


@pytest.fixture
def loan_checker():
    return LoanCheckerTool()


@pytest.fixture
def order_status():
    return OrderStatusTool()


@pytest.fixture
def recommendation_engine():
    return RecommendationEngineTool()


@pytest.fixture
def populated_registry():
    registry = ToolRegistry()
    registry.register(LoanCheckerTool())
    registry.register(OrderStatusTool())
    registry.register(RecommendationEngineTool())
    return registry


def test_wrap_as_langchain_tool_returns_structured_tool(loan_checker):
    """Test that wrap_as_langchain_tool returns a StructuredTool."""
    wrapped = wrap_as_langchain_tool(loan_checker)
    assert isinstance(wrapped, StructuredTool)


def test_wrapped_tool_has_correct_name(loan_checker):
    """Test that wrapped tool preserves the name."""
    wrapped = wrap_as_langchain_tool(loan_checker)
    assert wrapped.name == "loan_checker"


def test_wrapped_tool_has_correct_description(loan_checker):
    """Test that wrapped tool preserves the description."""
    wrapped = wrap_as_langchain_tool(loan_checker)
    assert wrapped.description == loan_checker.description


def test_wrapped_tool_has_args_schema(loan_checker):
    """Test that wrapped tool has a Pydantic args_schema."""
    wrapped = wrap_as_langchain_tool(loan_checker)
    assert wrapped.args_schema is not None
    assert hasattr(wrapped.args_schema, 'model_fields')


def test_args_schema_has_correct_fields(loan_checker):
    """Test that args_schema contains the expected fields."""
    wrapped = wrap_as_langchain_tool(loan_checker)
    fields = wrapped.args_schema.model_fields
    assert "customer_id" in fields
    assert "annual_income" in fields
    assert "loan_amount" in fields


def test_args_schema_preserves_descriptions(loan_checker):
    """Test that field descriptions are preserved in args_schema."""
    wrapped = wrap_as_langchain_tool(loan_checker)
    fields = wrapped.args_schema.model_fields
    assert fields["customer_id"].description is not None
    assert fields["annual_income"].description is not None


@pytest.mark.asyncio
async def test_wrapped_tool_execute_success(loan_checker):
    """Test that wrapped tool can execute successfully."""
    wrapped = wrap_as_langchain_tool(loan_checker)
    result = await wrapped.ainvoke({
        "customer_id": "TEST-001",
        "annual_income": 100000,
        "loan_amount": 300000
    })
    assert result is not None
    assert isinstance(result, str)
    assert "eligible" in result.lower() or "True" in result


@pytest.mark.asyncio
async def test_wrapped_tool_execute_with_error(order_status):
    """Test that wrapped tool handles errors correctly."""
    wrapped = wrap_as_langchain_tool(order_status)
    result = await wrapped.ainvoke({"order_id": "ORD-INVALID"})
    assert "Error:" in result or "not found" in result.lower()


@pytest.mark.asyncio
async def test_wrapped_tool_pydantic_validation():
    """Test that Pydantic validates input before execution."""
    loan_checker = LoanCheckerTool()
    wrapped = wrap_as_langchain_tool(loan_checker)
    
    # Missing required fields should raise ValidationError
    with pytest.raises(Exception) as exc_info:
        await wrapped.ainvoke({"customer_id": "TEST-002"})
    assert "validation" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()


def test_wrap_tools_for_environment_banking(populated_registry):
    """Test wrapping tools for banking environment."""
    tools = wrap_tools_for_environment(populated_registry, "banking")
    tool_names = {t.name for t in tools}
    
    assert "loan_checker" in tool_names
    assert "recommendation_engine" in tool_names
    assert "order_status" not in tool_names
    assert len(tools) == 2


def test_wrap_tools_for_environment_retail(populated_registry):
    """Test wrapping tools for retail environment."""
    tools = wrap_tools_for_environment(populated_registry, "retail")
    tool_names = {t.name for t in tools}
    
    assert "order_status" in tool_names
    assert "recommendation_engine" in tool_names
    assert "loan_checker" not in tool_names
    assert len(tools) == 2


def test_wrap_tools_for_environment_returns_structured_tools(populated_registry):
    """Test that all wrapped tools are StructuredTools."""
    tools = wrap_tools_for_environment(populated_registry, "banking")
    for tool in tools:
        assert isinstance(tool, StructuredTool)
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, 'args_schema')


@pytest.mark.asyncio
async def test_wrapped_tools_from_environment_are_executable(populated_registry):
    """Test that tools wrapped via environment are executable."""
    tools = wrap_tools_for_environment(populated_registry, "banking")
    
    # Find the loan_checker tool
    loan_tool = next(t for t in tools if t.name == "loan_checker")
    
    result = await loan_tool.ainvoke({
        "customer_id": "ENV-TEST-001",
        "annual_income": 75000,
        "loan_amount": 200000
    })
    
    assert result is not None
    assert isinstance(result, str)


def test_multiple_tools_wrapped_independently(loan_checker, order_status):
    """Test that wrapping multiple tools works independently."""
    wrapped_loan = wrap_as_langchain_tool(loan_checker)
    wrapped_order = wrap_as_langchain_tool(order_status)
    
    assert wrapped_loan.name == "loan_checker"
    assert wrapped_order.name == "order_status"
    assert wrapped_loan.args_schema != wrapped_order.args_schema


def test_schema_conversion_handles_optional_fields(recommendation_engine):
    """Test that optional fields in schema are handled correctly."""
    wrapped = wrap_as_langchain_tool(recommendation_engine)
    fields = wrapped.args_schema.model_fields
    
    # customer_id is required
    assert "customer_id" in fields
    # category is optional (not in required list)
    assert "category" in fields


@pytest.mark.asyncio
async def test_wrapped_tool_with_dict_output(loan_checker):
    """Test that wrapped tool converts dict output to string."""
    wrapped = wrap_as_langchain_tool(loan_checker)
    result = await wrapped.ainvoke({
        "customer_id": "DICT-TEST",
        "annual_income": 80000,
        "loan_amount": 100000
    })
    
    # Result should be a string representation of the dict
    assert isinstance(result, str)
    assert "{" in result  # Should contain dict formatting
