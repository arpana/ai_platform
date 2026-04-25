from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    input: str
    environment: str = "banking"
    session_id: str | None = None
    trace_id: str | None = None


class AgentResponse(BaseModel):
    output: str
    trace_id: str
    environment: str
    tools_used: list[str] = Field(default_factory=list)
    rag_docs_used: int = 0
    latency_ms: float = 0.0


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    environment_scopes: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    tool_name: str
    output: Any = None
    error: str | None = None
    blocked: bool = False
    latency_ms: float = 0.0


class PolicyResult(BaseModel):
    allowed: bool
    reason: str = ""
    action: str = ""
    environment: str = ""


class RadarStatus(str, Enum):
    APPROVED = "approved"
    UNDER_REVIEW = "under_review"
    STOP = "stop"


class RadarEntry(BaseModel):
    name: str
    status: RadarStatus
    category: str = ""
    notes: str = ""


class LLMRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2048


class LLMResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
