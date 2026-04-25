from typing import TypedDict
from langchain_core.messages import BaseMessage

from ai_platform.core.models import ToolResult, PolicyResult


class AgentState(TypedDict):
    """
    State dictionary that flows through the LangGraph agent orchestrator.
    
    This state is passed between nodes in the ReAct loop and accumulates
    information as the agent reasons, calls tools, and generates responses.
    """
    
    messages: list[BaseMessage]
    """LangChain message history containing the conversation."""
    
    environment: str
    """Environment scope: 'banking' or 'retail'."""
    
    tool_results: list[ToolResult]
    """Accumulated tool execution results."""
    
    rag_context: list[str]
    """Retrieved document context (populated in Phase 4)."""
    
    policy_results: list[PolicyResult]
    """Policy evaluation results (populated in Phase 5)."""
    
    iteration_count: int
    """Number of reasoning iterations (safety cap to prevent infinite loops)."""
    
    trace_id: str
    """OpenTelemetry trace ID for correlation."""
    
    session_id: str | None
    """Optional session ID for conversation continuity."""
