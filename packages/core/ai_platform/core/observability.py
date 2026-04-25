"""
Observability utilities for tracing and logging.

Provides:
- LangChain callback handler for LLM tracing
- JSON logging configuration with trace correlation
- Helper functions for custom spans
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

from langchain_core.callbacks.base import BaseCallbackHandler
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


class TracingCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that creates OpenTelemetry spans for LLM calls.
    
    Automatically traces:
    - LLM invocations with model, prompt, and token usage
    - Tool executions
    - Chain executions
    """
    
    def __init__(self, tracer_name: str = "ai_platform.llm"):
        super().__init__()
        self.tracer = trace.get_tracer(tracer_name)
        self._spans: dict[str, trace.Span] = {}
    
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """Called when LLM starts."""
        run_id = kwargs.get("run_id", "unknown")
        span = self.tracer.start_span(
            "llm.call",
            attributes={
                "llm.model": serialized.get("id", ["unknown"])[-1],
                "llm.prompts": json.dumps(prompts[:2]),  # First 2 prompts for brevity
                "llm.prompt_count": len(prompts),
            },
        )
        self._spans[str(run_id)] = span
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM ends."""
        run_id = kwargs.get("run_id", "unknown")
        span = self._spans.pop(str(run_id), None)
        
        if span:
            # Extract token usage if available
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                if token_usage:
                    span.set_attribute("llm.token_usage.prompt", token_usage.get("prompt_tokens", 0))
                    span.set_attribute("llm.token_usage.completion", token_usage.get("completion_tokens", 0))
                    span.set_attribute("llm.token_usage.total", token_usage.get("total_tokens", 0))
            
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM errors."""
        run_id = kwargs.get("run_id", "unknown")
        span = self._spans.pop(str(run_id), None)
        
        if span:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()
    
    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Called when tool starts."""
        run_id = kwargs.get("run_id", "unknown")
        tool_name = serialized.get("name", "unknown_tool")
        
        span = self.tracer.start_span(
            f"tool.{tool_name}",
            attributes={
                "tool.name": tool_name,
                "tool.input": input_str[:500],  # Truncate for brevity
            },
        )
        self._spans[str(run_id)] = span
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when tool ends."""
        run_id = kwargs.get("run_id", "unknown")
        span = self._spans.pop(str(run_id), None)
        
        if span:
            span.set_attribute("tool.output", output[:500])  # Truncate
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when tool errors."""
        run_id = kwargs.get("run_id", "unknown")
        span = self._spans.pop(str(run_id), None)
        
        if span:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()
    
    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        """Called when chain starts."""
        run_id = kwargs.get("run_id", "unknown")
        chain_name = serialized.get("id", ["unknown"])[-1]
        
        span = self.tracer.start_span(
            f"chain.{chain_name}",
            attributes={"chain.name": chain_name},
        )
        self._spans[str(run_id)] = span
    
    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Called when chain ends."""
        run_id = kwargs.get("run_id", "unknown")
        span = self._spans.pop(str(run_id), None)
        
        if span:
            span.set_status(Status(StatusCode.OK))
            span.end()
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when chain errors."""
        run_id = kwargs.get("run_id", "unknown")
        span = self._spans.pop(str(run_id), None)
        
        if span:
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
            span.end()


@contextmanager
def trace_operation(name: str, attributes: dict[str, Any] | None = None) -> Iterator[trace.Span]:
    """
    Context manager for tracing an operation.
    
    Usage:
        with trace_operation("tool.execute", {"tool_name": "loan_checker"}):
            # ... operation code ...
    
    Args:
        name: Span name
        attributes: Optional attributes to set on the span
        
    Yields:
        The created span
    """
    tracer = trace.get_tracer("ai_platform")
    span = tracer.start_span(name)
    
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    
    start_time = time.perf_counter()
    
    try:
        yield span
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        span.set_attribute("duration_ms", round(elapsed_ms, 2))
        span.end()


def get_trace_context() -> dict[str, str]:
    """
    Get current trace context for logging correlation.
    
    Returns:
        Dict with trace_id and span_id if available
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        ctx = span.get_span_context()
        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    return {}


class TraceContextFilter(logging.Filter):
    """
    Logging filter that adds trace context to log records.
    
    Adds trace_id and span_id to every log record if available.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        trace_ctx = get_trace_context()
        record.trace_id = trace_ctx.get("trace_id", "")
        record.span_id = trace_ctx.get("span_id", "")
        return True
