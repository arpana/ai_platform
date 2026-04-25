import asyncio
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from ai_platform.agents import AgentState, rag_node
from ai_platform.rag.chromadb_wrapper import ChromaDBWrapper
from ai_platform.rag.retriever import RAGRetriever


@pytest.fixture
def base_state() -> AgentState:
    return {
        "messages": [HumanMessage(content="What are the benefits of a savings account?")],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "test-trace-rag",
        "session_id": None,
    }


@pytest.fixture
def real_retriever():
    wrapper = ChromaDBWrapper(ephemeral=True)
    retriever = RAGRetriever(wrapper)
    retriever.index_document(
        text="savings accounts offer competitive interest rates and security",
        doc_id="savings1",
        environment="banking",
    )
    retriever.index_document(
        text="retail product returns are accepted within 30 days",
        doc_id="retail1",
        environment="retail",
    )
    return retriever


def test_rag_node_returns_empty_context(base_state):
    result = asyncio.run(rag_node(base_state, retriever=None))
    assert result["rag_context"] == []


def test_rag_node_with_retriever_returns_docs(base_state, real_retriever):
    result = asyncio.run(rag_node(base_state, retriever=real_retriever))
    assert isinstance(result["rag_context"], list)
    assert len(result["rag_context"]) > 0
    assert all(isinstance(s, str) for s in result["rag_context"])


def test_rag_node_extracts_human_message_as_query(real_retriever):
    state: AgentState = {
        "messages": [
            AIMessage(content="I can help you."),
            HumanMessage(content="interest rates on savings"),
        ],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "test-extract",
        "session_id": None,
    }
    result = asyncio.run(rag_node(state, retriever=real_retriever))
    assert len(result["rag_context"]) > 0


def test_rag_node_with_no_human_message(real_retriever):
    state: AgentState = {
        "messages": [AIMessage(content="Hello")],
        "environment": "banking",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "test-no-human",
        "session_id": None,
    }
    result = asyncio.run(rag_node(state, retriever=real_retriever))
    assert result["rag_context"] == []


def test_rag_node_preserves_other_state(base_state, real_retriever):
    result = asyncio.run(rag_node(base_state, retriever=real_retriever))
    assert result["messages"] == base_state["messages"]
    assert result["environment"] == base_state["environment"]
    assert result["tool_results"] == base_state["tool_results"]
    assert result["policy_results"] == base_state["policy_results"]
    assert result["iteration_count"] == base_state["iteration_count"]
    assert result["trace_id"] == base_state["trace_id"]
    assert result["session_id"] == base_state["session_id"]


def test_rag_node_with_retail_environment(real_retriever):
    state: AgentState = {
        "messages": [HumanMessage(content="product returns policy")],
        "environment": "retail",
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": "test-retail",
        "session_id": None,
    }
    result = asyncio.run(rag_node(state, retriever=real_retriever))
    assert isinstance(result["rag_context"], list)
    assert result["environment"] == "retail"
    assert len(result["rag_context"]) > 0


def test_rag_node_is_async(base_state):
    import inspect

    assert inspect.iscoroutinefunction(rag_node)
    coro = rag_node(base_state)
    assert inspect.iscoroutine(coro)
    result = asyncio.run(coro)
    assert "rag_context" in result
