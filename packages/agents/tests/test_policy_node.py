import pytest
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

from ai_platform.agents import AgentState, pre_tool_policy_node, post_tool_policy_node
from ai_platform.core.models import ToolResult
from ai_platform.policy import PolicyEngine, PIIDetector


@pytest.fixture
def policy_engine():
    workspace_root = Path(__file__).parent.parent.parent.parent
    config_dir = workspace_root / "configs" / "policies"
    return PolicyEngine(config_dir=config_dir)


@pytest.fixture
def pii_detector():
    return PIIDetector()


@pytest.fixture
def sample_state():
    return {
        "messages": [HumanMessage(content="Check my loan eligibility")],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "test-trace-policy",
        "session_id": None,
    }


@pytest.fixture
def state_with_tool_calls():
    return {
        "messages": [
            HumanMessage(content="Check my loan"),
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
        "trace_id": "test-pre-policy",
        "session_id": None,
    }


@pytest.fixture
def state_with_tool_results():
    tool_result = ToolResult(
        tool_name="loan_checker",
        output={"eligible": True, "max_amount": 400000},
        error=None,
        blocked=False,
        latency_ms=45.2,
    )

    return {
        "messages": [HumanMessage(content="Check loan")],
        "environment": "banking",
        "tool_results": [tool_result],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 2,
        "trace_id": "test-post-policy",
        "session_id": None,
    }


def test_pre_tool_policy_node_no_tool_calls(sample_state, policy_engine):
    updated_state = pre_tool_policy_node(sample_state, policy_engine)

    assert "policy_results" in updated_state
    assert updated_state["policy_results"] == []
    assert isinstance(updated_state["policy_results"], list)


def test_pre_tool_policy_node_preserves_state(sample_state, policy_engine):
    updated_state = pre_tool_policy_node(sample_state, policy_engine)

    assert updated_state["messages"] == sample_state["messages"]
    assert updated_state["environment"] == sample_state["environment"]
    assert updated_state["tool_results"] == sample_state["tool_results"]
    assert updated_state["rag_context"] == sample_state["rag_context"]
    assert updated_state["iteration_count"] == sample_state["iteration_count"]
    assert updated_state["trace_id"] == sample_state["trace_id"]
    assert updated_state["session_id"] == sample_state["session_id"]


def test_pre_tool_policy_node_evaluates_tool_calls(state_with_tool_calls, policy_engine):
    updated_state = pre_tool_policy_node(state_with_tool_calls, policy_engine)

    # loan_checker is allowed in banking — should have 1 policy result
    assert len(updated_state["policy_results"]) == 1
    assert updated_state["policy_results"][0].action == "loan_checker"
    assert updated_state["policy_results"][0].allowed is True
    assert updated_state["messages"][-1].tool_calls[0]["name"] == "loan_checker"


def test_pre_tool_policy_node_denied_tool(policy_engine):
    state: AgentState = {
        "messages": [
            HumanMessage(content="Check order"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "order_status", "args": {"order_id": "ORD-001"}, "id": "call_denied"}
                ],
            ),
        ],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 1,
        "trace_id": "test-denied",
        "session_id": None,
    }

    updated_state = pre_tool_policy_node(state, policy_engine)

    assert len(updated_state["policy_results"]) == 1
    assert updated_state["policy_results"][0].action == "order_status"
    assert updated_state["policy_results"][0].allowed is False


def test_pre_tool_policy_node_banking_environment(sample_state, policy_engine):
    updated_state = pre_tool_policy_node(sample_state, policy_engine)

    assert updated_state["environment"] == "banking"


def test_pre_tool_policy_node_retail_environment(policy_engine):
    state: AgentState = {
        "messages": [HumanMessage(content="Check my order")],
        "environment": "retail",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "test-retail-policy",
        "session_id": None,
    }

    updated_state = pre_tool_policy_node(state, policy_engine)

    assert updated_state["environment"] == "retail"


def test_post_tool_policy_node_pii_disabled_passthrough(policy_engine, pii_detector):
    # retail has pii_enforcement.enabled=false
    state: AgentState = {
        "messages": [HumanMessage(content="Check order")],
        "environment": "retail",
        "tool_results": [
            ToolResult(
                tool_name="order_status",
                output="Customer SSN: 123-45-6789",
                error=None,
                blocked=False,
                latency_ms=10.0,
            )
        ],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 1,
        "trace_id": "test-post-retail",
        "session_id": None,
    }

    updated_state = post_tool_policy_node(state, policy_engine, pii_detector)

    # PII enforcement disabled in retail — output should be unchanged
    assert updated_state is state


def test_post_tool_policy_node_sanitizes_pii_in_banking(policy_engine, pii_detector):
    # banking has pii_enforcement.enabled=true
    state: AgentState = {
        "messages": [HumanMessage(content="Check loan")],
        "environment": "banking",
        "tool_results": [
            ToolResult(
                tool_name="loan_checker",
                output="Customer SSN: 123-45-6789, Card: 4532-1234-5678-9010",
                error=None,
                blocked=False,
                latency_ms=20.0,
            )
        ],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 2,
        "trace_id": "test-post-banking",
        "session_id": None,
    }

    updated_state = post_tool_policy_node(state, policy_engine, pii_detector)

    sanitized_output = updated_state["tool_results"][0].output
    assert "123-45-6789" not in sanitized_output
    assert "[REDACTED_SSN]" in sanitized_output
    assert "4532-1234-5678-9010" not in sanitized_output
    assert "[REDACTED_CC]" in sanitized_output


def test_post_tool_policy_node_preserves_all_fields(policy_engine, pii_detector):
    state: AgentState = {
        "messages": [HumanMessage(content="Query"), AIMessage(content="Response")],
        "environment": "retail",
        "tool_results": [
            ToolResult(
                tool_name="test_tool",
                output="clean result",
                error=None,
                blocked=False,
                latency_ms=10.0,
            )
        ],
        "rag_context": ["doc1", "doc2"],
        "policy_results": [],
        "iteration_count": 3,
        "trace_id": "test-preserve",
        "session_id": "session-456",
    }

    updated_state = post_tool_policy_node(state, policy_engine, pii_detector)

    assert updated_state["messages"] == state["messages"]
    assert updated_state["environment"] == state["environment"]
    assert updated_state["rag_context"] == state["rag_context"]
    assert updated_state["iteration_count"] == state["iteration_count"]
    assert updated_state["trace_id"] == state["trace_id"]
    assert updated_state["session_id"] == state["session_id"]


def test_policy_nodes_are_deterministic(sample_state, policy_engine, pii_detector):
    pre_result1 = pre_tool_policy_node(sample_state, policy_engine)
    pre_result2 = pre_tool_policy_node(sample_state, policy_engine)

    assert pre_result1["policy_results"] == pre_result2["policy_results"]

    # Use retail env (PII disabled) so post_policy is a passthrough
    retail_state: AgentState = {**sample_state, "environment": "retail"}
    post_result1 = post_tool_policy_node(retail_state, policy_engine, pii_detector)
    post_result2 = post_tool_policy_node(retail_state, policy_engine, pii_detector)

    assert post_result1 == post_result2


def test_pre_policy_accumulates_policy_results(policy_engine):
    from ai_platform.core.models import PolicyResult

    existing_result = PolicyResult(
        allowed=True, reason="previous", action="old_tool", environment="banking"
    )

    state: AgentState = {
        "messages": [
            HumanMessage(content="Query"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "loan_checker",
                        "args": {
                            "customer_id": "C1",
                            "annual_income": 50000,
                            "loan_amount": 100000,
                        },
                        "id": "call_acc",
                    }
                ],
            ),
        ],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [existing_result],
        "iteration_count": 0,
        "trace_id": "test-accumulate",
        "session_id": None,
    }

    updated_state = pre_tool_policy_node(state, policy_engine)

    # Should have the original result plus the new one
    assert len(updated_state["policy_results"]) == 2
    assert updated_state["policy_results"][0] is existing_result
    assert updated_state["policy_results"][1].action == "loan_checker"
