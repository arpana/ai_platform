# Phase 7 - Observability

## Overview
Phase 7 implements comprehensive observability with OpenTelemetry tracing and structured JSON logging, providing visibility into agent execution, LLM calls, tool usage, RAG retrieval, and policy enforcement.

## Components Implemented

### 1. OpenTelemetry Setup (7.1)
**Location**: `packages/core/ai_platform/core/tracing.py`

- **Provider setup**: Integrated into FastAPI lifespan in `services/api/main.py`
- **Exporters**: 
  - Console exporter (development)
  - OTLP gRPC exporter (production)
- **Configuration**: Via environment variables (`AIP_OTEL_ENABLED`, `AIP_OTEL_EXPORTER`, `AIP_OTEL_ENDPOINT`)
- **Service metadata**: Includes service name and environment tags

### 2. LangChain Callback Handler (7.2)
**Location**: `packages/core/ai_platform/core/observability.py`

The `TracingCallbackHandler` automatically traces:
- **LLM calls**: Model name, prompts, token usage (prompt/completion/total)
- **Tool executions**: Tool name, input/output
- **Chain executions**: Chain names and flow
- **Errors**: Exception recording with stack traces

**Integration**: Wired into ChatOpenAI in `services/api/routes/agent.py`:
```python
tracing_callback = TracingCallbackHandler()
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[tracing_callback])
```

### 3. Custom Spans (7.3)

#### Tool Execution Spans
**Location**: `packages/agents/ai_platform/agents/graph.py`

Wraps each tool execution with:
- Span name: `tool.{tool_name}`
- Attributes: tool name, input (truncated), duration

#### RAG Retrieval Spans
**Location**: `packages/agents/ai_platform/agents/nodes/rag_node.py`

Traces document retrieval with:
- Span name: `rag.retrieve`
- Attributes: query (truncated), environment, docs count

#### Policy Check Spans
**Location**: `packages/agents/ai_platform/agents/nodes/policy_node.py`

Two policy operations traced:

**Pre-tool policy**:
- Span name: `policy.pre_check`
- Attributes: environment, tool count, allowed/denied counts

**PII sanitization**:
- Span name: `policy.pii_sanitize`
- Attributes: environment, PII instances found, results processed

### 4. Structured JSON Logging (7.4)
**Location**: `services/api/main.py`

- **Format**: JSON via `python-json-logger`
- **Trace correlation**: Automatic trace_id and span_id injection via `TraceContextFilter`
- **Development mode**: Human-readable format when `debug=true`
- **Production mode**: JSON format with all trace context

**Example log entry**:
```json
{
  "asctime": "2026-04-05 18:28:33,306",
  "levelname": "INFO",
  "name": "demo",
  "message": "Executing tool: loan_checker",
  "trace_id": "0xefbd61a7c4a5e922ea610e7338544a5d",
  "span_id": "0xfe5e15af1fc828b9",
  "tool_name": "loan_checker"
}
```

## Configuration

Environment variables:
```bash
AIP_OTEL_ENABLED=true              # Enable OpenTelemetry
AIP_OTEL_EXPORTER=console          # console or otlp
AIP_OTEL_ENDPOINT=http://localhost:4317  # OTLP endpoint
AIP_LOG_LEVEL=INFO                 # Logging level
AIP_DEBUG=false                    # Use JSON logs in production
```

## Testing

**Test suite**: `packages/core/tests/test_observability.py` (10 tests)

Coverage:
- TracingCallbackHandler creation and lifecycle
- LLM start/end/error callbacks
- trace_operation context manager
- get_trace_context function
- TraceContextFilter for logging

**Demo**: `scripts/observability_demo.py`

Shows complete workflow with:
- Agent execution span
- RAG retrieval tracing
- Policy checks (pre-check and PII sanitization)
- Tool execution
- Structured JSON logs with trace correlation

## Dependencies Added

`packages/core/pyproject.toml`:
```toml
"opentelemetry-api>=1.20.0"
"opentelemetry-sdk>=1.20.0"
"opentelemetry-exporter-otlp-proto-grpc>=1.20.0"
"python-json-logger>=2.0.0"
"langchain-core>=0.1.0"
```

## Span Attributes Reference

### RAG Spans
- `rag.query`: User query (truncated to 200 chars)
- `rag.environment`: Environment name
- `rag.docs_count`: Number of documents retrieved
- `duration_ms`: Operation duration

### Policy Spans
- `policy.environment`: Environment name
- `policy.tool_count`: Number of tools checked
- `policy.allowed_count`: Tools allowed
- `policy.denied_count`: Tools denied
- `pii.enabled`: PII enforcement enabled
- `pii.instances_found`: PII patterns found
- `pii.results_processed`: Results sanitized

### Tool Spans
- `tool.name`: Tool identifier
- `tool.input`: Tool input (truncated to 500 chars)
- `tool.output`: Tool output (truncated, if successful)
- `duration_ms`: Execution time

### LLM Spans
- `llm.model`: Model name (e.g., "gpt-4o-mini")
- `llm.prompts`: First 2 prompts
- `llm.prompt_count`: Total prompts
- `llm.token_usage.prompt`: Prompt tokens
- `llm.token_usage.completion`: Completion tokens
- `llm.token_usage.total`: Total tokens

## Usage Examples

### Custom Span Creation
```python
from ai_platform.core.observability import trace_operation

with trace_operation("custom.operation", {"key": "value"}) as span:
    # Your operation
    result = await some_async_operation()
    span.set_attribute("result_count", len(result))
```

### Getting Trace Context
```python
from ai_platform.core.observability import get_trace_context

trace_ctx = get_trace_context()
logger.info("Processing request", extra={
    "trace_id": trace_ctx.get("trace_id", ""),
    "custom_field": "value",
})
```

### Using Tracing Callback
```python
from ai_platform.core.observability import TracingCallbackHandler
from langchain_openai import ChatOpenAI

callback = TracingCallbackHandler()
llm = ChatOpenAI(model="gpt-4", callbacks=[callback])
```

## Integration with Monitoring

The OTLP exporter can send traces to:
- **Jaeger**: Set `AIP_OTEL_ENDPOINT=http://jaeger:4317`
- **Grafana Tempo**: Set endpoint to Tempo OTLP receiver
- **Datadog**: Use Datadog OTLP endpoint
- **New Relic**: Configure New Relic OTLP endpoint

## Test Results

**Phase 5-7 Combined**: 125 tests passing
- 21 radar tests
- 37 policy tests
- 51 agent tests (with tracing integration)
- 10 observability tests
- 6 integration tests

## Benefits

1. **Full visibility**: Trace every operation from API request to LLM call
2. **Performance monitoring**: Duration tracking for all operations
3. **Error tracking**: Automatic exception recording in spans
4. **Log correlation**: Trace IDs link logs to distributed traces
5. **Production-ready**: JSON logs + OTLP export for observability platforms
6. **Development-friendly**: Console exporter for local debugging
