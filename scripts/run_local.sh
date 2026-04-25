#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONDA_ENV="ai-platform"
HOST="${AIP_HOST:-0.0.0.0}"
PORT="${AIP_PORT:-8000}"
RELOAD="${AIP_RELOAD:-true}"

check_conda_env() {
    if ! conda info --envs 2>/dev/null | grep -q "^${CONDA_ENV} "; then
        echo "Conda environment '${CONDA_ENV}' not found."
        echo "Create it with: conda env create -f ${PROJECT_ROOT}/environment.yml"
        exit 1
    fi
}

install_packages() {
    echo "Installing local packages in editable mode..."
    pip install -e "${PROJECT_ROOT}/packages/core"
    pip install -e "${PROJECT_ROOT}/packages/kairos"
    pip install -e "${PROJECT_ROOT}/packages/tools"
    pip install -e "${PROJECT_ROOT}/packages/rag"
    pip install -e "${PROJECT_ROOT}/packages/policy"
    pip install -e "${PROJECT_ROOT}/packages/radar"
    pip install -e "${PROJECT_ROOT}/packages/agents"
    echo "All packages installed."
}

start_server() {
    echo "Starting AI Platform API on ${HOST}:${PORT}..."
    cd "$PROJECT_ROOT"

    if [ "$RELOAD" = "true" ]; then
        uvicorn services.api.main:app --host "$HOST" --port "$PORT" --reload
    else
        uvicorn services.api.main:app --host "$HOST" --port "$PORT"
    fi
}

case "${1:-start}" in
    install)
        check_conda_env
        install_packages
        ;;
    start)
        start_server
        ;;
    setup)
        check_conda_env
        install_packages
        start_server
        ;;
    *)
        echo "Usage: $0 {install|start|setup}"
        echo ""
        echo "  install  - Install all local packages (editable mode)"
        echo "  start    - Start the FastAPI server"
        echo "  setup    - Install packages + start server"
        exit 1
        ;;
esac
