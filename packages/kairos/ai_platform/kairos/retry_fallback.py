from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from typing import Any

from ai_platform.core.exceptions import KairosError
from ai_platform.core.models import LLMRequest, LLMResponse

from .interface import BaseLLMProvider

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func: Callable[[], Coroutine[Any, Any, LLMResponse]],
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> LLMResponse:
    delay = initial_delay

    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except KairosError:
            if attempt == max_attempts:
                raise
            jitter_offset = random.uniform(-0.1, 0.1) * delay if jitter else 0.0
            sleep_time = delay + jitter_offset
            logger.warning(
                "Attempt %d/%d failed. Retrying in %.1fs",
                attempt,
                max_attempts,
                sleep_time,
            )
            await asyncio.sleep(sleep_time)
            delay *= backoff_factor

    raise KairosError("Retry exhausted without result", code="RETRY_EXHAUSTED")


class FallbackProvider(BaseLLMProvider):
    def __init__(self, providers: list[BaseLLMProvider]) -> None:
        self._providers = providers

    async def complete(self, request: LLMRequest) -> LLMResponse:
        errors: list[Exception] = []

        for provider in self._providers:
            try:
                return await retry_with_backoff(
                    lambda p=provider: p.complete(request),
                )
            except Exception as e:
                errors.append(e)
                logger.warning(
                    "Provider %s failed: %s",
                    type(provider).__name__,
                    e,
                )

        raise KairosError(
            f"All {len(self._providers)} providers failed. Last: {errors[-1]}",
            code="ALL_PROVIDERS_FAILED",
        )
