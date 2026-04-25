from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ai_platform.agents import AgentState, create_agent_graph
from ai_platform.core.config import EnvironmentConfig
from ai_platform.core.models import AgentRequest, AgentResponse
from ai_platform.policy import PolicyEngine, PIIDetector
from ai_platform.radar import TechRadar
from ai_platform.rag.chromadb_wrapper import ChromaDBWrapper
from ai_platform.rag.retriever import RAGRetriever
from ai_platform.tools import (
    ToolRegistry,
    LoanCheckerTool,
    OrderStatusTool,
    RecommendationEngineTool,
)
from services.api.dependencies import get_env_config

router = APIRouter(prefix="/agent", tags=["agent"])

# Find workspace root (services/api/routes/agent.py -> ../../../../)
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent


def get_rag_retriever() -> RAGRetriever:
    wrapper = ChromaDBWrapper(persist_dir="./data/chroma")
    return RAGRetriever(wrapper)


def get_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(LoanCheckerTool())
    registry.register(OrderStatusTool())
    registry.register(RecommendationEngineTool())
    return registry


def get_llm() -> ChatOpenAI:
    """
    Get or create the LLM instance.

    Uses OPENAI_API_KEY or AIP_KAIROS_API_KEY from environment.
    """
    from ai_platform.core.observability import TracingCallbackHandler

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIP_KAIROS_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY or AIP_KAIROS_API_KEY must be set")

    # Create LLM with tracing callback
    tracing_callback = TracingCallbackHandler()
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=api_key,
        callbacks=[tracing_callback],
    )


def get_policy_engine() -> PolicyEngine:
    """
    Get or create the PolicyEngine instance.

    Loads policy configurations from configs/policies/ directory.
    """
    config_dir = WORKSPACE_ROOT / "configs" / "policies"
    return PolicyEngine(config_dir=config_dir)


def get_pii_detector() -> PIIDetector:
    """
    Get or create the PIIDetector instance.

    Uses default PII patterns (credit cards, SSN, emails, phones).
    """
    return PIIDetector()


def get_radar_registry() -> TechRadar:
    """
    Get or create the TechRadar instance.

    Loads tech radar from configs/radar/tech_radar.yaml.
    """
    config_path = WORKSPACE_ROOT / "configs" / "radar" / "tech_radar.yaml"
    return TechRadar(config_path=config_path)


@router.post("/execute", response_model=AgentResponse)
async def execute_agent(
    request: AgentRequest,
    env_config: EnvironmentConfig = Depends(get_env_config),
    registry: ToolRegistry = Depends(get_tool_registry),
    llm: ChatOpenAI = Depends(get_llm),
    retriever: RAGRetriever = Depends(get_rag_retriever),
    policy_engine: PolicyEngine = Depends(get_policy_engine),
    pii_detector: PIIDetector = Depends(get_pii_detector),
    radar_registry: TechRadar = Depends(get_radar_registry),
):
    """
    Execute the agent with the given request.

    This endpoint:
    1. Builds initial AgentState from the request
    2. Compiles the LangGraph ReAct agent with policy enforcement
    3. Invokes the graph to process the request
    4. Extracts results and returns AgentResponse
    """
    start = time.perf_counter()
    trace_id = request.trace_id or uuid.uuid4().hex

    # Build initial AgentState
    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.input)],
        "environment": env_config.name,
        "tool_results": [],
        "rag_context": [],
        "policy_results": [],
        "iteration_count": 0,
        "trace_id": trace_id,
        "session_id": request.session_id,
    }

    # Create and compile the LangGraph agent with policy enforcement
    graph = create_agent_graph(
        llm=llm,
        registry=registry,
        policy_engine=policy_engine,
        pii_detector=pii_detector,
        radar_registry=radar_registry,
        retriever=retriever,
    )

    # Invoke the graph
    final_state = await graph.ainvoke(initial_state)

    # Extract the final AI message
    final_message = final_state["messages"][-1]
    output = final_message.content if hasattr(final_message, "content") else str(final_message)

    # Extract tool names used
    tools_used = list(set(result.tool_name for result in final_state["tool_results"]))

    # Count RAG docs (will be > 0 in Phase 4)
    rag_docs_used = len(final_state["rag_context"])

    # Calculate latency
    elapsed = (time.perf_counter() - start) * 1000

    return AgentResponse(
        output=output,
        trace_id=trace_id,
        environment=env_config.name,
        tools_used=tools_used,
        rag_docs_used=rag_docs_used,
        latency_ms=round(elapsed, 2),
    )
