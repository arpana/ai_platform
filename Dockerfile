# =============================================================================
# Stage 1: builder — install all dependencies
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build-time system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install FastAPI runtime dependencies first (for better layer caching)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pyyaml \
    pydantic \
    pydantic-settings \
    python-dotenv \
    langchain \
    langgraph \
    langchain-openai \
    langchain-core \
    langchain-community \
    chromadb \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-grpc \
    python-json-logger

# Copy local packages in dependency order
COPY packages/core /app/packages/core
RUN pip install --no-cache-dir -e /app/packages/core

COPY packages/kairos /app/packages/kairos
RUN pip install --no-cache-dir -e /app/packages/kairos

COPY packages/tools /app/packages/tools
RUN pip install --no-cache-dir -e /app/packages/tools

COPY packages/rag /app/packages/rag
RUN pip install --no-cache-dir -e /app/packages/rag

COPY packages/policy /app/packages/policy
RUN pip install --no-cache-dir -e /app/packages/policy

COPY packages/radar /app/packages/radar
RUN pip install --no-cache-dir -e /app/packages/radar

COPY packages/agents /app/packages/agents
RUN pip install --no-cache-dir -e /app/packages/agents

# =============================================================================
# Stage 2: runtime — lean production image
# =============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install minimal runtime system deps (e.g. for chromadb native libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --no-create-home appuser

# Copy installed site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application source
COPY --from=builder /app/packages /app/packages
COPY services /app/services
COPY configs /app/configs

# Create ChromaDB data dir and set permissions
RUN mkdir -p /app/data/chroma \
    && chown -R appuser:appuser /app

EXPOSE 8000

USER appuser

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
