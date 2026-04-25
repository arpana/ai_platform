from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pythonjsonlogger import jsonlogger

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_platform.core.config import get_settings
from ai_platform.core.exceptions import PlatformError, PolicyViolationError, RadarBlockedError
from ai_platform.core.tracing import setup_tracing
from ai_platform.core.observability import TraceContextFilter
from services.api.routes import agent, health, radar

logger = logging.getLogger("ai_platform")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Setup structured JSON logging with trace correlation
    log_handler = logging.StreamHandler()
    
    if settings.debug or settings.log_level.upper() == "DEBUG":
        # Human-readable format for development
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    else:
        # JSON format for production
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s %(span_id)s",
            timestamp=True,
        )
        log_handler.setFormatter(json_formatter)
        log_handler.addFilter(TraceContextFilter())
    
    logging.root.handlers = [log_handler]
    logging.root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    logger.info("Starting AI Platform", extra={
        "environment": settings.environment,
        "otel_enabled": settings.otel_enabled,
    })

    # Setup OpenTelemetry tracing
    setup_tracing(settings)

    yield

    logger.info("Shutting down AI Platform")


app = FastAPI(
    title="Agentic AI Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(agent.router)
app.include_router(radar.router)


@app.exception_handler(PolicyViolationError)
async def policy_violation_handler(request: Request, exc: PolicyViolationError):
    return JSONResponse(
        status_code=403,
        content={
            "error": "policy_violation",
            "message": exc.message,
            "tool": exc.tool_name,
            "environment": exc.environment,
        },
    )


@app.exception_handler(RadarBlockedError)
async def radar_blocked_handler(request: Request, exc: RadarBlockedError):
    return JSONResponse(
        status_code=403,
        content={
            "error": "radar_blocked",
            "message": exc.message,
            "tool": exc.tool_name,
            "status": exc.status,
        },
    )


@app.exception_handler(PlatformError)
async def platform_error_handler(request: Request, exc: PlatformError):
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.code or "platform_error",
            "message": exc.message,
        },
    )
