from __future__ import annotations

import logging

from ai_platform.core.config import Settings, get_settings, load_environment_config
from ai_platform.core.exceptions import KairosError
from ai_platform.core.models import LLMRequest, LLMResponse

from .interface import BaseLLMProvider
from .providers.mock import MockProvider
from .providers.openai import OpenAIProvider
from .retry_fallback import FallbackProvider, retry_with_backoff

logger = logging.getLogger(__name__)


class KairosAdapter:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._provider = self._build_provider()
        logger.info(
            "KairosAdapter initialised: provider=%s, model=%s",
            self._settings.kairos_provider,
            self._settings.kairos_model,
        )

    def _build_provider(self) -> BaseLLMProvider:
        name = self._settings.kairos_provider

        if name == "mock":
            return MockProvider()

        if name == "openai":
            return OpenAIProvider(api_key=self._settings.kairos_api_key)

        if name == "fallback":
            return FallbackProvider(
                [
                    OpenAIProvider(api_key=self._settings.kairos_api_key),
                    MockProvider(),
                ]
            )

        raise KairosError(
            f"Unknown kairos provider: {name}",
            code="UNKNOWN_PROVIDER",
        )

    @property
    def provider(self) -> BaseLLMProvider:
        return self._provider

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await retry_with_backoff(lambda: self._provider.complete(request))

    async def complete_for_environment(
        self,
        request: LLMRequest,
        environment: str,
    ) -> LLMResponse:
        env_config = load_environment_config(environment)
        patched = request.model_copy(update={"model": env_config.model})
        return await self.complete(patched)
