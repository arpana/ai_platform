"""
Full integration tests for the ReAct agent graph.

These tests verify the complete agent workflow with mocked LLM and tools:
- Full graph execution from start to end
- Multi-turn reasoning with tool calls
- Error handling and recovery
- Iteration limits
- Environment-specific tool filtering
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from ai_platform.agents import AgentState, create_agent_graph, reasoning_node, MAX_ITERATIONS
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
def policy_engine():
    """Create a PolicyEngine instance."""
    from pathlib import Path
    # Point to the workspace root configs/policies directory
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_dir = workspace_root / "configs" / "policies"
    return PolicyEngine(config_dir=config_dir)


@pytest.fixture
def pii_detector():
    """Create a PIIDetector instance."""
    return PIIDetector()


@pytest.fixture
def radar_registry():
    """Create a TechRadar instance."""
    from pathlib import Path
    # Point to the workspace root configs/radar directory
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_path = workspace_root / "configs" / "radar" / "tech_radar.yaml"
    return TechRadar(config_path=config_path)


@pytest.fixture
def mock_llm_no_tools():
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=AIMessage(content="I can help you with that!"))
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=bound)
    return mock


@pytest.fixture
def mock_llm_with_tool_call():
    call_count = {"n": 0}

    async def side_effect(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return AIMessage(
                content="I'll check that for you.",
                tool_calls=[
                    {
                        "name": "loan_checker",
                        "args": {
                            "customer_id": "CUST-001",
                            "annual_income": 80000,
                            "loan_amount": 200000,
                        },
                        "id": "call_mock_123",
                    }
                ],
            )
        return AIMessage(content="Based on the results, you are eligible for the loan!")

    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=side_effect)
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=bound)
    return mock


@pytest.fixture
def mock_llm_multi_turn():
    call_count = {"n": 0}

    async def side_effect(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "loan_checker",
                        "args": {
                            "customer_id": "CUST-002",
                            "annual_income": 100000,
                            "loan_amount": 300000,
                        },
                        "id": "call_1",
                    }
                ],
            )
        elif call_count["n"] == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "recommendation_engine",
                        "args": {"customer_id": "CUST-002", "environment": "banking"},
                        "id": "call_2",
                    }
                ],
            )
        return AIMessage(content="Here are your results and recommendations!")

    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=side_effect)
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=bound)
    return mock


class TestGraphIntegration:
    """Integration tests for the full agent graph."""

    @pytest.mark.asyncio
    async def test_graph_simple_response(self, mock_llm_no_tools, registry, policy_engine, pii_detector, radar_registry):
        """Test graph with a simple response (no tool calls)."""
        graph = create_agent_graph(mock_llm_no_tools, registry, policy_engine, pii_detector, radar_registry)

        initial_state: AgentState = {
            "messages": [HumanMessage(content="Hello, how are you?")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "test-simple",
            "session_id": None,
        }

        final_state = await graph.ainvoke(initial_state)

        # Should have 2 messages: Human + AI
        assert len(final_state["messages"]) == 2
        assert isinstance(final_state["messages"][0], HumanMessage)
        assert isinstance(final_state["messages"][1], AIMessage)

        # Should have exactly 1 iteration
        assert final_state["iteration_count"] == 1

        # No tools should be called
        assert len(final_state["tool_results"]) == 0

    @pytest.mark.asyncio
    async def test_graph_with_tool_call(self, mock_llm_with_tool_call, registry, policy_engine, pii_detector, radar_registry):
        """Test graph with one tool call followed by response."""
        graph = create_agent_graph(mock_llm_with_tool_call, registry, policy_engine, pii_detector, radar_registry)

        initial_state: AgentState = {
            "messages": [HumanMessage(content="Check my loan eligibility")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "test-tool",
            "session_id": None,
        }

        final_state = await graph.ainvoke(initial_state)

        # Should have 4 messages: Human + AI (tool call) + Tool + AI (final)
        assert len(final_state["messages"]) == 4

        # Verify message types
        assert isinstance(final_state["messages"][0], HumanMessage)
        assert isinstance(final_state["messages"][1], AIMessage)
        assert isinstance(final_state["messages"][2], ToolMessage)
        assert isinstance(final_state["messages"][3], AIMessage)

        # Should have 2 iterations
        assert final_state["iteration_count"] == 2

        # One tool should be called
        assert len(final_state["tool_results"]) == 1
        assert final_state["tool_results"][0].tool_name == "loan_checker"
        assert final_state["tool_results"][0].error is None  # Success means no error

    @pytest.mark.asyncio
    async def test_graph_multi_turn(self, mock_llm_multi_turn, registry, policy_engine, pii_detector, radar_registry):
        """Test graph with multiple tool calls across iterations."""
        graph = create_agent_graph(mock_llm_multi_turn, registry, policy_engine, pii_detector, radar_registry)

        initial_state: AgentState = {
            "messages": [HumanMessage(content="Check my loan and give recommendations")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "test-multi",
            "session_id": None,
        }

        final_state = await graph.ainvoke(initial_state)

        # Should have multiple messages
        assert len(final_state["messages"]) >= 5  # Human + AI + Tool + AI + Tool + AI

        # Should have 3 iterations
        assert final_state["iteration_count"] == 3

        # Two tools should be called
        assert len(final_state["tool_results"]) == 2
        tool_names = [result.tool_name for result in final_state["tool_results"]]
        assert "loan_checker" in tool_names
        assert "recommendation_engine" in tool_names

    @pytest.mark.asyncio
    async def test_graph_max_iterations(self, registry, policy_engine, pii_detector, radar_registry):
        """Test that graph stops at max iterations."""
        bound = MagicMock()
        bound.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "loan_checker",
                        "args": {
                            "customer_id": "CUST-999",
                            "annual_income": 50000,
                            "loan_amount": 100000,
                        },
                        "id": "call_infinite",
                    }
                ],
            )
        )
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=bound)

        graph = create_agent_graph(mock_llm, registry, policy_engine, pii_detector, radar_registry)

        initial_state: AgentState = {
            "messages": [HumanMessage(content="Keep checking loans")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "test-max",
            "session_id": None,
        }

        final_state = await graph.ainvoke(initial_state)

        # Should stop at MAX_ITERATIONS
        assert final_state["iteration_count"] == MAX_ITERATIONS

        # Should have MAX_ITERATIONS - 1 tool results
        # (Last iteration increments count but doesn't execute tool due to should_continue check)
        assert len(final_state["tool_results"]) == MAX_ITERATIONS - 1

    @pytest.mark.asyncio
    async def test_graph_retail_environment(self, mock_llm_no_tools, registry, policy_engine, pii_detector, radar_registry):
        """Test graph respects retail environment."""
        graph = create_agent_graph(mock_llm_no_tools, registry, policy_engine, pii_detector, radar_registry)

        initial_state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "environment": "retail",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "test-retail",
            "session_id": None,
        }

        final_state = await graph.ainvoke(initial_state)

        # Environment should be preserved
        assert final_state["environment"] == "retail"

    @pytest.mark.asyncio
    async def test_graph_preserves_context(self, mock_llm_no_tools, registry, policy_engine, pii_detector, radar_registry):
        """Test that graph preserves all state fields."""
        graph = create_agent_graph(mock_llm_no_tools, registry, policy_engine, pii_detector, radar_registry)

        initial_state: AgentState = {
            "messages": [HumanMessage(content="Test")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": ["Some RAG context"],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "trace-123",
            "session_id": "session-456",
        }

        final_state = await graph.ainvoke(initial_state)

        # All context should be preserved
        assert final_state["rag_context"] == ["Some RAG context"]
        assert final_state["trace_id"] == "trace-123"
        assert final_state["session_id"] == "session-456"


class TestReasoningNodeMocked:
    """Tests for reasoning_node with mocked LLM."""

    @pytest.mark.asyncio
    async def test_reasoning_node_increments_count(self, mock_llm_no_tools, registry):
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 5,
            "trace_id": "test",
            "session_id": None,
        }

        updated_state = await reasoning_node(state, mock_llm_no_tools, registry)

        assert updated_state["iteration_count"] == 6

    @pytest.mark.asyncio
    async def test_reasoning_node_adds_ai_message(self, mock_llm_no_tools, registry):
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "test",
            "session_id": None,
        }

        updated_state = await reasoning_node(state, mock_llm_no_tools, registry)

        assert len(updated_state["messages"]) == 2
        assert isinstance(updated_state["messages"][1], AIMessage)

    @pytest.mark.asyncio
    async def test_reasoning_node_with_tool_call(self, mock_llm_with_tool_call, registry):
        state: AgentState = {
            "messages": [HumanMessage(content="Check my loan")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "test",
            "session_id": None,
        }

        updated_state = await reasoning_node(state, mock_llm_with_tool_call, registry)

        assert len(updated_state["messages"]) == 2
        ai_message = updated_state["messages"][1]
        assert isinstance(ai_message, AIMessage)
        assert len(ai_message.tool_calls) == 1
        assert ai_message.tool_calls[0]["name"] == "loan_checker"

    @pytest.mark.asyncio
    async def test_reasoning_node_filters_tools_by_environment(self, registry):
        captured_tools = None

        def mock_bind_tools(tools):
            nonlocal captured_tools
            captured_tools = tools
            bound = MagicMock()
            bound.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))
            return bound

        mock_llm = MagicMock()
        mock_llm.bind_tools = mock_bind_tools

        state: AgentState = {
            "messages": [HumanMessage(content="Test")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "test",
            "session_id": None,
        }

        await reasoning_node(state, mock_llm, registry)

        assert captured_tools is not None
        assert len(captured_tools) > 0

    @pytest.mark.asyncio
    async def test_reasoning_node_preserves_state(self, mock_llm_no_tools, registry):
        state: AgentState = {
            "messages": [HumanMessage(content="Test")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": ["Context 1"],
            "policy_results": [],
            "iteration_count": 3,
            "trace_id": "trace-789",
            "session_id": "session-123",
        }

        updated_state = await reasoning_node(state, mock_llm_no_tools, registry)

        assert updated_state["environment"] == "banking"
        assert updated_state["rag_context"] == ["Context 1"]
        assert updated_state["trace_id"] == "trace-789"
        assert updated_state["session_id"] == "session-123"
        assert updated_state["tool_results"] == []


class TestErrorHandling:
    """Tests for error handling in the graph."""

    @pytest.mark.asyncio
    async def test_graph_handles_tool_error(self, registry, policy_engine, pii_detector, radar_registry):
        """Test that graph handles tool execution errors gracefully."""
        call_count = {"n": 0}

        async def side_effect(messages):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "loan_checker",
                            "args": {"customer_id": "CUST-001"},
                            "id": "call_error",
                        }
                    ],
                )
            return AIMessage(content="I encountered an error but handled it.")

        bound = MagicMock()
        bound.ainvoke = AsyncMock(side_effect=side_effect)
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=bound)

        graph = create_agent_graph(mock_llm, registry, policy_engine, pii_detector, radar_registry)

        initial_state: AgentState = {
            "messages": [HumanMessage(content="Check my loan")],
            "environment": "banking",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "test-error",
            "session_id": None,
        }

        final_state = await graph.ainvoke(initial_state)

        # Graph should complete despite error
        assert len(final_state["messages"]) >= 3

        # Tool result should indicate error
        assert len(final_state["tool_results"]) == 1
        # Error is recorded but execution continues

    @pytest.mark.asyncio
    async def test_graph_with_empty_messages(self, mock_llm_no_tools, registry, policy_engine, pii_detector, radar_registry):
        """Test graph behavior with invalid initial state."""
        graph = create_agent_graph(mock_llm_no_tools, registry, policy_engine, pii_detector, radar_registry)

        initial_state: AgentState = {
            "messages": [],
            "environment": "banking",
            "tool_results": [],
            "rag_context": [],
            "policy_results": [],
            "iteration_count": 0,
            "trace_id": "test-empty",
            "session_id": None,
        }

        # Should handle empty messages gracefully
        # (behavior depends on should_continue logic - it returns "end")
        final_state = await graph.ainvoke(initial_state)

        # Should not crash
        assert final_state is not None
