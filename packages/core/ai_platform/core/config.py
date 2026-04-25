from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from ai_platform.core.exceptions import ConfigurationError

_CONFIGS_DIR = Path(__file__).resolve().parents[4] / "configs"


class EnvironmentConfig(BaseModel):
    name: str
    tools: list[str] = Field(default_factory=list)
    pii: str = "relaxed"
    rag_collection: str = ""
    model: str = "gpt-4o-mini"
    max_agent_steps: int = 20


class Settings(BaseSettings):
    environment: str = "banking"
    log_level: str = "INFO"
    debug: bool = False

    kairos_provider: str = "mock"
    kairos_api_key: str = ""
    kairos_model: str = "gpt-4o-mini"

    chroma_persist_dir: str = "./data/chroma"

    otel_enabled: bool = False
    otel_exporter: str = "console"
    otel_endpoint: str = "http://localhost:4317"

    model_config = {"env_prefix": "AIP_"}


def load_environment_config(environment: str, configs_dir: Path | None = None) -> EnvironmentConfig:
    base_dir = configs_dir or _CONFIGS_DIR
    config_path = base_dir / "environments" / f"{environment}.yaml"

    if not config_path.exists():
        raise ConfigurationError(
            f"Environment config not found: {config_path}",
            code="CONFIG_NOT_FOUND",
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return EnvironmentConfig(name=environment, **raw)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"YAML file not found: {path}", code="FILE_NOT_FOUND")
    with open(path) as f:
        return yaml.safe_load(f) or {}
