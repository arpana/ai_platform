from __future__ import annotations

from fastapi import APIRouter, Depends

from ai_platform.core.config import EnvironmentConfig, Settings
from services.api.dependencies import get_app_settings, get_env_config

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    return {"status": "healthy"}


@router.get("/ready")
def readiness_check(
    settings: Settings = Depends(get_app_settings),
    env_config: EnvironmentConfig = Depends(get_env_config),
):
    return {
        "status": "ready",
        "environment": env_config.name,
        "tools_registered": len(env_config.tools),
        "kairos_provider": settings.kairos_provider,
    }
