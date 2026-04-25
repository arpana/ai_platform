from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

if TYPE_CHECKING:
    from ai_platform.core.config import Settings

logger = logging.getLogger(__name__)

_initialized = False


def setup_tracing(settings: Settings) -> TracerProvider | None:
    global _initialized
    if _initialized or not settings.otel_enabled:
        return None

    resource = Resource.create(
        {"service.name": "ai-platform", "deployment.environment": settings.environment}
    )
    provider = TracerProvider(resource=resource)

    if settings.otel_exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif settings.otel_exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            logger.warning("OTLP exporter not installed, falling back to console")
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _initialized = True
    logger.info("OpenTelemetry tracing initialized (exporter=%s)", settings.otel_exporter)
    return provider


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
