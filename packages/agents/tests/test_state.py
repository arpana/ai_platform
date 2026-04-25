import pytest
from langchain_core.messages import HumanMessage, AIMessage
from ai_platform.agents import AgentState
from ai_platform.core.models import ToolResult, PolicyResult


def test_agent_state_structure():
    """Test that AgentState has all required fields."""
    state: AgentState = {
        "messages": [HumanMessage(content="Hello")],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "test-trace-123",
        "session_id": None
    }
    
    assert state["environment"] == "banking"
    assert len(state["messages"]) == 1
    assert state["iteration_count"] == 0
    assert state["trace_id"] == "test-trace-123"


def test_agent_state_with_messages():
    """Test AgentState with multiple messages."""
    state: AgentState = {
        "messages": [
            HumanMessage(content="What's my loan eligibility?"),
            AIMessage(content="I'll check that for you.")
        ],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 1,
        "trace_id": "trace-456",
        "session_id": "session-abc"
    }
    
    assert len(state["messages"]) == 2
    assert state["session_id"] == "session-abc"
    assert state["iteration_count"] == 1


def test_agent_state_with_tool_results():
    """Test AgentState with accumulated tool results."""
    tool_result = ToolResult(
        tool_name="loan_checker",
        output={"eligible": True, "max_amount": 500000},
        error=None,
        blocked=False,
        latency_ms=45.2
    )
    
    state: AgentState = {
        "messages": [HumanMessage(content="Check my loan")],
        "environment": "banking",
        "tool_results": [tool_result],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 2,
        "trace_id": "trace-789",
        "session_id": None
    }
    
    assert len(state["tool_results"]) == 1
    assert state["tool_results"][0].tool_name == "loan_checker"
    assert state["tool_results"][0].output["eligible"] is True


def test_agent_state_with_rag_context():
    """Test AgentState with RAG context (Phase 4 preparation)."""
    state: AgentState = {
        "messages": [HumanMessage(content="Tell me about savings accounts")],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [
            "Document 1: Savings accounts offer 4.5% APY",
            "Document 2: Minimum balance requirement is $1000"
        ],
        "policy_results": [],
        "iteration_count": 1,
        "trace_id": "trace-rag",
        "session_id": None
    }
    
    assert len(state["rag_context"]) == 2
    assert "4.5% APY" in state["rag_context"][0]


def test_agent_state_with_policy_results():
    """Test AgentState with policy results (Phase 5 preparation)."""
    policy_result = PolicyResult(
        allowed=True,
        reason="Tool is permitted in banking environment",
        action="loan_checker",
        environment="banking"
    )
    
    state: AgentState = {
        "messages": [HumanMessage(content="Check eligibility")],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [policy_result],
        "iteration_count": 1,
        "trace_id": "trace-policy",
        "session_id": None
    }
    
    assert len(state["policy_results"]) == 1
    assert state["policy_results"][0].allowed is True
    assert state["policy_results"][0].environment == "banking"


def test_agent_state_retail_environment():
    """Test AgentState with retail environment."""
    state: AgentState = {
        "messages": [HumanMessage(content="Check my order")],
        "environment": "retail",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "trace-retail",
        "session_id": "retail-session-123"
    }
    
    assert state["environment"] == "retail"
    assert state["session_id"] == "retail-session-123"


def test_agent_state_iteration_safety():
    """Test AgentState tracks iteration count for safety cap."""
    state: AgentState = {
        "messages": [HumanMessage(content="Complex query")],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 5,
        "trace_id": "trace-iteration",
        "session_id": None
    }
    
    assert state["iteration_count"] == 5
    # In the actual graph, we'd check if iteration_count >= MAX_ITERATIONS


def test_agent_state_empty_lists():
    """Test AgentState with empty accumulator lists."""
    state: AgentState = {
        "messages": [],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "trace-empty",
        "session_id": None
    }
    
    assert state["messages"] == []
    assert state["tool_results"] == []
    assert state["rag_context"] == []
    assert state["policy_results"] == []
