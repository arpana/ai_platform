"""
Shared fixtures for E2E tests.

Provides:
- Environment variable setup with cache clearing
- FastAPI TestClient
- Mock LLM factories (no real OpenAI calls)
- Mock RAG retriever (no ChromaDB needed)
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from ai_platform.core.config import get_settings
from services.api.main import app


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    monkeypatch.setenv("AIP_KAIROS_API_KEY", "sk-test-key-e2e")
    monkeypatch.setenv("AIP_KAIROS_PROVIDER", "openai")
    monkeypatch.setenv("AIP_ENVIRONMENT", "banking")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_rag_retriever():
    from ai_platform.rag.retriever import RAGRetriever

    mock = MagicMock(spec=RAGRetriever)
    mock.retrieve.return_value = []
    return mock


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
