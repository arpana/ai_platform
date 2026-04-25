from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_platform.rag.retriever import RAGRetriever

from langchain_core.messages import HumanMessage

from ai_platform.agents.state import AgentState
from ai_platform.core.observability import trace_operation

logger = logging.getLogger(__name__)


async def rag_node(state: AgentState, retriever: RAGRetriever | None = None) -> AgentState:
    """
    RAG retrieval node - retrieves relevant documents from vector store.
    
    Args:
        state: Current agent state
        retriever: Optional RAGRetriever instance
        
    Returns:
        Updated state with rag_context populated
    """
    if retriever is None:
        return state

    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {**state, "rag_context": []}

    query = human_messages[-1].content
    environment = state["environment"]
    
    # Trace RAG retrieval
    with trace_operation(
        "rag.retrieve",
        attributes={
            "rag.query": query[:200],  # Truncate for brevity
            "rag.environment": environment,
        }
    ) as span:
        docs = retriever.retrieve(query=query, environment=environment)
        span.set_attribute("rag.docs_count", len(docs))
    
    return {**state, "rag_context": docs}
