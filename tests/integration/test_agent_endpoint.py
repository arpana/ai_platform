"""
Integration tests for the /agent/execute endpoint.

These tests verify the full stack integration:
- FastAPI endpoint → LangGraph agent → Tools → LLM → Response
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from services.api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_env():
    """Set up environment variables for testing."""
    os.environ["AIP_KAIROS_API_KEY"] = "sk-test-key-123"
    os.environ["AIP_KAIROS_PROVIDER"] = "openai"
    os.environ["AIP_ENVIRONMENT"] = "banking"
    yield
    # Cleanup is optional since these are test-specific


class TestAgentEndpoint:
    """Test suite for the /agent/execute endpoint."""

    @patch("services.api.routes.agent.create_agent_graph")
    def test_endpoint_exists(self, mock_create_graph, client):
        """Test that the endpoint exists and accepts POST requests."""
        # Mock the graph to avoid real LLM calls
        mock_graph = MagicMock()
        
        async def mock_ainvoke(state):
            return {
                **state,
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi there!"),
                ],
                "tool_results": [],
                "rag_context": [],
                "policy_results": [],
                "iteration_count": 1,
            }
        
        mock_graph.ainvoke.side_effect = mock_ainvoke
        mock_create_graph.return_value = mock_graph
        
        response = client.post(
            "/agent/execute",
            json={"input": "Hello", "environment": "banking"},
        )
        # Should not return 404 or 405
        assert response.status_code != 404
        assert response.status_code != 405
        assert response.status_code == 200

    def test_request_validation(self, client):
        """Test that the endpoint validates the request body."""
        # Missing required field 'input'
        response = client.post("/agent/execute", json={"environment": "banking"})
        assert response.status_code == 422  # Validation error

    @patch("services.api.routes.agent.create_agent_graph")
    def test_agent_execution_mock(self, mock_create_graph, client):
        """Test agent execution with a mocked graph."""
        # Mock the graph to return a simple response
        mock_graph = MagicMock()
        mock_graph.ainvoke = MagicMock()
        
        # Create a mock final state
        async def mock_ainvoke(state):
            return {
                **state,
                "messages": [
                    HumanMessage(content="What is my loan status?"),
                    AIMessage(content="Your loan is approved."),
                ],
                "tool_results": [],
                "rag_context": [],
                "policy_results": [],
                "iteration_count": 1,
            }
        
        mock_graph.ainvoke.side_effect = mock_ainvoke
        mock_create_graph.return_value = mock_graph

        # Make the request
        response = client.post(
            "/agent/execute",
            json={
                "input": "What is my loan status?",
                "environment": "banking",
            },
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        
        assert "output" in data
        assert "trace_id" in data
        assert "environment" in data
        assert "tools_used" in data
        assert "rag_docs_used" in data
        assert "latency_ms" in data
        
        assert data["environment"] == "banking"
        assert isinstance(data["tools_used"], list)
        assert isinstance(data["rag_docs_used"], int)
        assert data["latency_ms"] > 0

    @patch("services.api.routes.agent.create_agent_graph")
    def test_agent_with_tools(self, mock_create_graph, client):
        """Test agent execution that uses tools."""
        # Mock the graph to return a response with tool usage
        mock_graph = MagicMock()
        mock_graph.ainvoke = MagicMock()
        
        async def mock_ainvoke(state):
            from ai_platform.core.models import ToolResult
            
            return {
                **state,
                "messages": [
                    HumanMessage(content="Check my loan status"),
                    AIMessage(content="I'll check your loan status."),
                    AIMessage(content="Your loan is approved with 5% interest."),
                ],
                "tool_results": [
                    ToolResult(
                        tool_name="loan_checker",
                        success=True,
                        output={"status": "approved", "interest_rate": 0.05},
                        error=None,
                    )
                ],
                "rag_context": [],
                "policy_results": [],
                "iteration_count": 2,
            }
        
        mock_graph.ainvoke.side_effect = mock_ainvoke
        mock_create_graph.return_value = mock_graph

        # Make the request
        response = client.post(
            "/agent/execute",
            json={
                "input": "Check my loan status",
                "environment": "banking",
            },
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        
        assert data["tools_used"] == ["loan_checker"]
        assert data["rag_docs_used"] == 0

    def test_trace_id_generation(self, client):
        """Test that trace_id is generated if not provided."""
        with patch("services.api.routes.agent.create_agent_graph") as mock_create_graph:
            mock_graph = MagicMock()
            
            async def mock_ainvoke(state):
                return {
                    **state,
                    "messages": [
                        HumanMessage(content="Test"),
                        AIMessage(content="Response"),
                    ],
                    "tool_results": [],
                    "rag_context": [],
                    "policy_results": [],
                    "iteration_count": 1,
                }
            
            mock_graph.ainvoke.side_effect = mock_ainvoke
            mock_create_graph.return_value = mock_graph

            # Request without trace_id
            response1 = client.post(
                "/agent/execute",
                json={"input": "Test 1", "environment": "banking"},
            )
            
            # Request with trace_id
            response2 = client.post(
                "/agent/execute",
                json={
                    "input": "Test 2",
                    "environment": "banking",
                    "trace_id": "custom-trace-123",
                },
            )

            assert response1.status_code == 200
            assert response2.status_code == 200
            
            # Auto-generated trace_id should be present
            assert response1.json()["trace_id"]
            # Custom trace_id should be preserved
            assert response2.json()["trace_id"] == "custom-trace-123"

    def test_environment_header(self, client):
        """Test that environment is correctly passed through."""
        with patch("services.api.routes.agent.create_agent_graph") as mock_create_graph:
            mock_graph = MagicMock()
            
            async def mock_ainvoke(state):
                return {
                    **state,
                    "messages": [
                        HumanMessage(content="Test"),
                        AIMessage(content="Response"),
                    ],
                    "tool_results": [],
                    "rag_context": [],
                    "policy_results": [],
                    "iteration_count": 1,
                }
            
            mock_graph.ainvoke.side_effect = mock_ainvoke
            mock_create_graph.return_value = mock_graph

            # Test retail environment
            response = client.post(
                "/agent/execute",
                json={"input": "Test", "environment": "retail"},
            )

            assert response.status_code == 200
            assert response.json()["environment"] == "retail"
