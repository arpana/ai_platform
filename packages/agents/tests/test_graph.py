import pytest
import os
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from ai_platform.agents import (
    AgentState,
    create_agent_graph,
    reasoning_node,
    tool_node,
    should_continue,
    MAX_ITERATIONS,
)
from ai_platform.tools import (
    ToolRegistry,
    LoanCheckerTool,
    OrderStatusTool,
    RecommendationEngineTool,
)
from ai_platform.policy import PolicyEngine, PIIDetector
from ai_platform.radar import TechRadar


@pytest.fixture
def registry():
    """Create a tool registry with mock tools."""
    reg = ToolRegistry()
    reg.register(LoanCheckerTool())
    reg.register(OrderStatusTool())
    reg.register(RecommendationEngineTool())
    return reg


@pytest.fixture
def llm():
    """Create a ChatOpenAI instance for testing."""
    api_key = os.getenv("AIP_KAIROS_API_KEY", "test-key")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


@pytest.fixture
def policy_engine():
    """Create a PolicyEngine with config directory."""
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_dir = workspace_root / "configs" / "policies"
    return PolicyEngine(config_dir=config_dir)


@pytest.fixture
def pii_detector():
    """Create a PIIDetector with default patterns."""
    return PIIDetector()


@pytest.fixture
def radar_registry():
    """Create a TechRadar instance."""
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_path = workspace_root / "configs" / "radar" / "tech_radar.yaml"
    return TechRadar(config_path=config_path)


@pytest.fixture
def initial_state():
    """Create an initial agent state."""
    return {
        "messages": [HumanMessage(content="Hello")],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "test-trace-123",
        "session_id": None,
    }


def test_should_continue_with_tool_calls():
    """Test should_continue returns 'tools' when last message has tool_calls."""
    state: AgentState = {
        "messages": [
            HumanMessage(content="Check loan"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "loan_checker",
                        "args": {
                            "customer_id": "123",
                            "annual_income": 80000,
                            "loan_amount": 200000,
                        },
                        "id": "call_123",
                    }
                ],
            ),
        ],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 1,
        "trace_id": "test",
        "session_id": None,
    }

    result = should_continue(state)
    assert result == "tools"


def test_should_continue_without_tool_calls():
    """Test should_continue returns 'end' when last message has no tool_calls."""
    state: AgentState = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hello! How can I help you?"),
        ],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 1,
        "trace_id": "test",
        "session_id": None,
    }

    result = should_continue(state)
    assert result == "end"


def test_should_continue_max_iterations():
    """Test should_continue returns 'end' when max iterations reached."""
    state: AgentState = {
        "messages": [
            HumanMessage(content="Check loan"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "loan_checker",
                        "args": {
                            "customer_id": "123",
                            "annual_income": 80000,
                            "loan_amount": 200000,
                        },
                        "id": "call_123",
                    }
                ],
            ),
        ],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": MAX_ITERATIONS,
        "trace_id": "test",
        "session_id": None,
    }

    result = should_continue(state)
    assert result == "end"


def test_should_continue_empty_messages():
    """Test should_continue handles empty messages list."""
    state: AgentState = {
        "messages": [],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "test",
        "session_id": None,
    }

    result = should_continue(state)
    assert result == "end"


@pytest.mark.asyncio
async def test_tool_node_executes_tool(registry):
    """Test tool_node executes tools and updates state."""
    state: AgentState = {
        "messages": [
            HumanMessage(content="Check my loan eligibility"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "loan_checker",
                        "args": {
                            "customer_id": "CUST-001",
                            "annual_income": 100000,
                            "loan_amount": 300000,
                        },
                        "id": "call_loan_123",
                    }
                ],
            ),
        ],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 1,
        "trace_id": "test",
        "session_id": None,
    }

    updated_state = await tool_node(state, registry)

    # Check that tool message was added
    assert len(updated_state["messages"]) == 3
    last_message = updated_state["messages"][-1]
    assert isinstance(last_message, ToolMessage)
    assert last_message.name == "loan_checker"

    # Check that tool result was recorded
    assert len(updated_state["tool_results"]) == 1
    assert updated_state["tool_results"][0].tool_name == "loan_checker"


@pytest.mark.asyncio
async def test_tool_node_handles_error(registry):
    """Test tool_node handles tool errors gracefully."""
    state: AgentState = {
        "messages": [
            HumanMessage(content="Check my loan"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "loan_checker",
                        "args": {
                            "customer_id": "CUST-002"
                            # Missing required fields
                        },
                        "id": "call_loan_error",
                    }
                ],
            ),
        ],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 1,
        "trace_id": "test",
        "session_id": None,
    }

    updated_state = await tool_node(state, registry)

    # Check that error message was added
    assert len(updated_state["messages"]) == 3
    last_message = updated_state["messages"][-1]
    assert isinstance(last_message, ToolMessage)
    assert "Error" in last_message.content or "Missing" in last_message.content


@pytest.mark.asyncio
async def test_tool_node_multiple_tools(registry):
    """Test tool_node executes multiple tool calls."""
    state: AgentState = {
        "messages": [
            HumanMessage(content="Check loan and get recommendations"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "loan_checker",
                        "args": {
                            "customer_id": "CUST-001",
                            "annual_income": 80000,
                            "loan_amount": 200000,
                        },
                        "id": "call_loan",
                    },
                    {
                        "name": "recommendation_engine",
                        "args": {"customer_id": "CUST-001", "environment": "banking"},
                        "id": "call_rec",
                    },
                ],
            ),
        ],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 1,
        "trace_id": "test",
        "session_id": None,
    }

    updated_state = await tool_node(state, registry)

    # Check that two tool messages were added
    assert len(updated_state["messages"]) == 4  # Human + AI + 2 ToolMessages
    assert isinstance(updated_state["messages"][-1], ToolMessage)
    assert isinstance(updated_state["messages"][-2], ToolMessage)

    # Check that two tool results were recorded
    assert len(updated_state["tool_results"]) == 2


def test_create_agent_graph(llm, registry, policy_engine, pii_detector, radar_registry):
    """Test that create_agent_graph creates a compiled graph."""
    graph = create_agent_graph(llm, registry, policy_engine, pii_detector, radar_registry)

    # Verify it's a compiled graph
    assert graph is not None
    assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")


def test_graph_structure(llm, registry):
    """Test that the graph has the expected structure."""
    from langgraph.graph import StateGraph

    # Create uncompiled workflow to inspect structure
    workflow = StateGraph(AgentState)
    workflow.add_node("reasoning", lambda state: reasoning_node(state, llm, registry))
    workflow.add_node("tools", lambda state: tool_node(state, registry))
    workflow.set_entry_point("reasoning")

    # Verify nodes were added
    assert "reasoning" in workflow.nodes
    assert "tools" in workflow.nodes


def test_max_iterations_constant():
    """Test that MAX_ITERATIONS is set correctly."""
    assert MAX_ITERATIONS == 15
    assert isinstance(MAX_ITERATIONS, int)
    assert MAX_ITERATIONS > 0


@pytest.mark.asyncio
async def test_reasoning_node_increments_iteration(llm, registry, initial_state):
    if not os.getenv("AIP_KAIROS_API_KEY"):
        pytest.skip("No API key available")

    initial_count = initial_state["iteration_count"]
    updated_state = await reasoning_node(initial_state, llm, registry)

    assert updated_state["iteration_count"] == initial_count + 1


@pytest.mark.asyncio
async def test_reasoning_node_adds_message(llm, registry, initial_state):
    if not os.getenv("AIP_KAIROS_API_KEY"):
        pytest.skip("No API key available")

    initial_message_count = len(initial_state["messages"])
    updated_state = await reasoning_node(initial_state, llm, registry)

    assert len(updated_state["messages"]) == initial_message_count + 1
    assert isinstance(updated_state["messages"][-1], AIMessage)
