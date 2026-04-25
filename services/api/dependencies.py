from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Request

from ai_platform.core.config import (
    EnvironmentConfig,
    Settings,
    get_settings,
    load_environment_config,
)

if TYPE_CHECKING:
    pass


def get_app_settings() -> Settings:
    return get_settings()


def get_env_config(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> EnvironmentConfig:
    environment = request.headers.get("X-Environment", settings.environment)
    return load_environment_config(environment)
