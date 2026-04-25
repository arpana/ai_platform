from __future__ import annotations

import time

from ai_platform.core.models import LLMRequest, LLMResponse

from ..interface import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.monotonic()
        content = f"[MOCK] Response to: {request.prompt[:200]}"
        latency_ms = (time.monotonic() - start) * 1000

        return LLMResponse(
            content=content,
            model=request.model,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(content.split()),
            latency_ms=latency_ms,
        )
