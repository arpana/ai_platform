"""
Tests for observability utilities.

Tests tracing callback handler and trace helpers.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from ai_platform.core.observability import (
    TracingCallbackHandler,
    trace_operation,
    get_trace_context,
    TraceContextFilter,
)


class TestTracingCallbackHandler:
    """Test LangChain callback handler for tracing."""
    
    def test_callback_handler_creation(self):
        """Test creating a callback handler."""
        handler = TracingCallbackHandler()
        assert handler is not None
        assert handler.tracer is not None
    
    @patch("ai_platform.core.observability.trace.get_tracer")
    def test_on_llm_start(self, mock_get_tracer):
        """Test LLM start callback creates span."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer
        
        handler = TracingCallbackHandler()
        handler.on_llm_start(
            serialized={"id": ["langchain", "ChatOpenAI"]},
            prompts=["Hello"],
            run_id="test-run-id"
        )
        
        # Verify span was created and stored
        assert "test-run-id" in handler._spans
        mock_tracer.start_span.assert_called_once()
    
    @patch("ai_platform.core.observability.trace.get_tracer")
    def test_on_llm_end(self, mock_get_tracer):
        """Test LLM end callback closes span."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        
        handler = TracingCallbackHandler()
        handler._spans["test-run-id"] = mock_span
        
        response = MagicMock()
        response.llm_output = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        
        handler.on_llm_end(response, run_id="test-run-id")
        
        # Verify span was set and ended
        mock_span.set_attribute.assert_called()
        mock_span.set_status.assert_called_once()
        mock_span.end.assert_called_once()
        assert "test-run-id" not in handler._spans
    
    @patch("ai_platform.core.observability.trace.get_tracer")
    def test_on_llm_error(self, mock_get_tracer):
        """Test LLM error callback records exception."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_get_tracer.return_value = mock_tracer
        
        handler = TracingCallbackHandler()
        handler._spans["test-run-id"] = mock_span
        
        error = ValueError("Test error")
        handler.on_llm_error(error, run_id="test-run-id")
        
        # Verify error was recorded
        mock_span.record_exception.assert_called_once_with(error)
        mock_span.end.assert_called_once()
        assert "test-run-id" not in handler._spans


class TestTraceOperation:
    """Test trace_operation context manager."""
    
    @patch("ai_platform.core.observability.trace.get_tracer")
    def test_trace_operation_success(self, mock_get_tracer):
        """Test successful traced operation."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer
        
        with trace_operation("test.operation", {"key": "value"}):
            pass
        
        # Verify span was created and ended
        mock_tracer.start_span.assert_called_once_with("test.operation")
        mock_span.set_attribute.assert_called()
        mock_span.end.assert_called_once()
    
    @patch("ai_platform.core.observability.trace.get_tracer")
    def test_trace_operation_error(self, mock_get_tracer):
        """Test traced operation with error."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer
        
        with pytest.raises(ValueError):
            with trace_operation("test.operation"):
                raise ValueError("Test error")
        
        # Verify error was recorded
        mock_span.record_exception.assert_called_once()
        mock_span.end.assert_called_once()


class TestGetTraceContext:
    """Test get_trace_context function."""
    
    @patch("ai_platform.core.observability.trace.get_current_span")
    def test_get_trace_context_with_span(self, mock_get_current_span):
        """Test getting trace context with active span."""
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.is_valid = True
        mock_ctx.trace_id = 12345
        mock_ctx.span_id = 67890
        mock_span.get_span_context.return_value = mock_ctx
        mock_get_current_span.return_value = mock_span
        
        context = get_trace_context()
        
        assert "trace_id" in context
        assert "span_id" in context
    
    @patch("ai_platform.core.observability.trace.get_current_span")
    def test_get_trace_context_without_span(self, mock_get_current_span):
        """Test getting trace context without active span."""
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.is_valid = False
        mock_span.get_span_context.return_value = mock_ctx
        mock_get_current_span.return_value = mock_span
        
        context = get_trace_context()
        
        assert context == {}


class TestTraceContextFilter:
    """Test logging filter for trace context."""
    
    @patch("ai_platform.core.observability.get_trace_context")
    def test_filter_adds_trace_context(self, mock_get_trace_context):
        """Test filter adds trace context to log record."""
        mock_get_trace_context.return_value = {
            "trace_id": "abc123",
            "span_id": "def456",
        }
        
        log_filter = TraceContextFilter()
        record = MagicMock()
        
        result = log_filter.filter(record)
        
        assert result is True
        assert record.trace_id == "abc123"
        assert record.span_id == "def456"
    
    @patch("ai_platform.core.observability.get_trace_context")
    def test_filter_with_no_context(self, mock_get_trace_context):
        """Test filter with no active trace context."""
        mock_get_trace_context.return_value = {}
        
        log_filter = TraceContextFilter()
        record = MagicMock()
        
        result = log_filter.filter(record)
        
        assert result is True
        assert record.trace_id == ""
        assert record.span_id == ""
