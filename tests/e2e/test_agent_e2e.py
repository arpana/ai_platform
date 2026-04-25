from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from services.api.main import app
from services.api.routes.agent import get_llm, get_rag_retriever


def make_simple_mock_llm(content: str = "I can help you."):
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=bound)
    return mock


def make_tool_call_mock_llm(tool_name: str, tool_args: dict, final_content: str = "Done."):
    call_count = {"n": 0}

    async def side_effect(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                    }
                ],
            )
        return AIMessage(content=final_content)

    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=side_effect)
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=bound)
    return mock


@pytest.fixture
def client():
    return TestClient(app)


def test_banking_simple_response(client, mock_rag_retriever):
    mock_llm = make_simple_mock_llm("I can help with banking.")
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_rag_retriever] = lambda: mock_rag_retriever

    response = client.post(
        "/agent/execute",
        json={"input": "Tell me about banking services.", "environment": "banking"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["output"]
    assert data["tools_used"] == []
    assert data["environment"] == "banking"


def test_banking_with_loan_checker_tool(client, mock_rag_retriever):
    mock_llm = make_tool_call_mock_llm(
        tool_name="loan_checker",
        tool_args={"customer_id": "CUST-001", "annual_income": 80000, "loan_amount": 200000},
        final_content="Loan approved!",
    )
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_rag_retriever] = lambda: mock_rag_retriever

    response = client.post(
        "/agent/execute",
        json={"input": "Check loan for customer CUST-001", "environment": "banking"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "loan_checker" in data["tools_used"]
    assert "Loan" in data["output"]


def test_retail_simple_response(client, mock_rag_retriever):
    mock_llm = make_simple_mock_llm("I can help with retail.")
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_rag_retriever] = lambda: mock_rag_retriever

    response = client.post(
        "/agent/execute",
        json={"input": "What products do you have?", "environment": "retail"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["environment"] == "retail"
    assert data["tools_used"] == []


def test_retail_with_order_status_tool(client, mock_rag_retriever):
    mock_llm = make_tool_call_mock_llm(
        tool_name="order_status",
        tool_args={"order_id": "ORD-123"},
        final_content="Order is shipped!",
    )
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_rag_retriever] = lambda: mock_rag_retriever

    response = client.post(
        "/agent/execute",
        json={"input": "What is the status of order ORD-123?", "environment": "retail"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "order_status" in data["tools_used"]


def test_policy_blocks_banking_tool_in_retail(client, mock_rag_retriever):
    mock_llm = make_tool_call_mock_llm(
        tool_name="loan_checker",
        tool_args={"customer_id": "CUST-001", "annual_income": 50000, "loan_amount": 100000},
        final_content="Processed.",
    )
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_rag_retriever] = lambda: mock_rag_retriever

    response = client.post(
        "/agent/execute",
        json={"input": "Check loan eligibility", "environment": "retail"},
    )

    assert response.status_code in (200, 403, 500) and response.status_code != 500


def test_pii_sanitized_in_banking_response(client, mock_rag_retriever):
    mock_llm = make_tool_call_mock_llm(
        tool_name="loan_checker",
        tool_args={"customer_id": "CUST-SSN", "annual_income": 50000, "loan_amount": 100000},
        final_content="Customer application processed.",
    )
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_rag_retriever] = lambda: mock_rag_retriever

    response = client.post(
        "/agent/execute",
        json={"input": "Process loan for customer with SSN 123-45-6789", "environment": "banking"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "123-45-6789" not in data["output"]


def test_radar_blocks_stop_tool(client, mock_rag_retriever):
    mock_llm = make_tool_call_mock_llm(
        tool_name="external_pricing_api",
        tool_args={},
        final_content="Pricing fetched.",
    )
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_rag_retriever] = lambda: mock_rag_retriever

    response = client.post(
        "/agent/execute",
        json={"input": "Get pricing from external API", "environment": "retail"},
    )

    assert response.status_code == 403
    data = response.json()
    assert data["error"] == "radar_blocked"


def test_session_id_preserved(client, mock_rag_retriever):
    mock_llm = make_simple_mock_llm("Session response.")
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_rag_retriever] = lambda: mock_rag_retriever

    response = client.post(
        "/agent/execute",
        json={
            "input": "Hello",
            "environment": "banking",
            "session_id": "session-abc",
        },
    )

    assert response.status_code == 200


def test_custom_trace_id_in_response(client, mock_rag_retriever):
    mock_llm = make_simple_mock_llm("Trace response.")
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_rag_retriever] = lambda: mock_rag_retriever

    response = client.post(
        "/agent/execute",
        json={
            "input": "Hello",
            "environment": "banking",
            "trace_id": "e2e-test-trace",
        },
    )

    assert response.status_code == 200
    assert response.json()["trace_id"] == "e2e-test-trace"
