"""
Observability Demo - Phase 7

Demonstrates:
1. OpenTelemetry tracing with console output
2. LangChain callback handler for LLM tracing
3. Custom spans for tool execution, RAG, policy
4. Structured JSON logging with trace correlation

To run:
    AIP_OTEL_ENABLED=true python scripts/observability_demo.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables
os.environ["AIP_OTEL_ENABLED"] = "true"
os.environ["AIP_OTEL_EXPORTER"] = "console"
os.environ["AIP_LOG_LEVEL"] = "INFO"
os.environ["AIP_KAIROS_API_KEY"] = os.getenv("AIP_KAIROS_API_KEY", "demo-key")

from pythonjsonlogger import jsonlogger

from ai_platform.core.config import get_settings
from ai_platform.core.observability import (
    TraceContextFilter,
    TracingCallbackHandler,
    get_trace_context,
    trace_operation,
)
from ai_platform.core.tracing import setup_tracing

# Setup structured logging
log_handler = logging.StreamHandler()
json_formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s %(span_id)s",
    timestamp=True,
)
log_handler.setFormatter(json_formatter)
log_handler.addFilter(TraceContextFilter())

logging.root.handlers = [log_handler]
logging.root.setLevel(logging.INFO)

logger = logging.getLogger("demo")


async def simulate_tool_execution(tool_name: str):
    """Simulate a tool execution with tracing."""
    with trace_operation(
        f"tool.{tool_name}",
        attributes={
            "tool.name": tool_name,
            "tool.input": '{"customer_id": "CUST-123"}',
        },
    ):
        # Get current trace context
        trace_ctx = get_trace_context()
        logger.info(
            f"Executing tool: {tool_name}",
            extra={
                "tool_name": tool_name,
                "trace_id": trace_ctx.get("trace_id", ""),
            },
        )
        
        # Simulate async work
        await asyncio.sleep(0.1)
        
        logger.info(f"Tool {tool_name} completed successfully")


async def simulate_rag_retrieval(query: str):
    """Simulate RAG retrieval with tracing."""
    with trace_operation(
        "rag.retrieve",
        attributes={
            "rag.query": query[:200],
            "rag.environment": "banking",
        },
    ) as span:
        logger.info(f"Retrieving documents for: {query}")
        
        # Simulate retrieval
        await asyncio.sleep(0.05)
        docs_count = 3
        
        span.set_attribute("rag.docs_count", docs_count)
        logger.info(f"Retrieved {docs_count} documents")


async def simulate_policy_check(tool_name: str):
    """Simulate policy check with tracing."""
    with trace_operation(
        "policy.pre_check",
        attributes={
            "policy.environment": "banking",
            "policy.tool_count": 1,
        },
    ) as span:
        logger.info(f"Checking policy for tool: {tool_name}")
        
        # Simulate policy evaluation
        await asyncio.sleep(0.02)
        
        span.set_attribute("policy.allowed_count", 1)
        span.set_attribute("policy.denied_count", 0)
        logger.info(f"Policy check passed for {tool_name}")


async def main():
    """Main demo function."""
    settings = get_settings()
    
    print("\n" + "=" * 60)
    print("Phase 7 - Observability Demo")
    print("=" * 60)
    print(f"\nOpenTelemetry Enabled: {settings.otel_enabled}")
    print(f"Exporter: {settings.otel_exporter}")
    print(f"Log Level: {settings.log_level}")
    print("\n" + "=" * 60 + "\n")
    
    # Initialize tracing
    setup_tracing(settings)
    
    # Create a tracing callback handler (would be used with LangChain LLM)
    tracing_callback = TracingCallbackHandler()
    logger.info("Tracing callback handler created", extra={"handler": str(type(tracing_callback))})
    
    # Simulate an agent workflow
    with trace_operation("agent.execute", attributes={"environment": "banking"}):
        logger.info("Starting agent execution")
        
        # Simulate RAG retrieval
        await simulate_rag_retrieval("What is my loan status?")
        
        # Simulate policy check
        await simulate_policy_check("loan_checker")
        
        # Simulate tool execution
        await simulate_tool_execution("loan_checker")
        
        # Simulate policy check for PII
        with trace_operation(
            "policy.pii_sanitize",
            attributes={
                "policy.environment": "banking",
                "pii.enabled": True,
            },
        ) as span:
            logger.info("Sanitizing output for PII")
            await asyncio.sleep(0.02)
            
            span.set_attribute("pii.instances_found", 1)
            span.set_attribute("pii.results_processed", 1)
            logger.info("PII sanitization complete")
        
        logger.info("Agent execution completed")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    print("\nTrace spans shown above include:")
    print("- agent.execute (parent span)")
    print("- rag.retrieve (document retrieval)")
    print("- policy.pre_check (tool permission check)")
    print("- tool.loan_checker (tool execution)")
    print("- policy.pii_sanitize (PII sanitization)")
    print("\nLogs include trace_id and span_id for correlation")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
