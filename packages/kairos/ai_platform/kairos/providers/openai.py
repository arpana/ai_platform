from __future__ import annotations

import time

from langchain_openai import ChatOpenAI

from ai_platform.core.exceptions import KairosError
from ai_platform.core.models import LLMRequest, LLMResponse

from ..interface import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise KairosError(
                "API key is required for OpenAI provider",
                code="MISSING_API_KEY",
            )
        self._api_key = api_key
        self._clients: dict[str, ChatOpenAI] = {}

    def _get_client(self, request: LLMRequest) -> ChatOpenAI:
        cache_key = f"{request.model}:{request.temperature}:{request.max_tokens}"
        if cache_key not in self._clients:
            self._clients[cache_key] = ChatOpenAI(
                api_key=self._api_key,
                model=request.model,
                temperature=request.temperature,
                max_completion_tokens=request.max_tokens,
            )
        return self._clients[cache_key]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            client = self._get_client(request)
            start = time.monotonic()
            response = await client.ainvoke(request.prompt)
            latency_ms = (time.monotonic() - start) * 1000

            # langchain-core >= 0.2 exposes usage_metadata on AIMessage
            usage = getattr(response, "usage_metadata", None) or {}
            prompt_tokens = (
                usage.get("input_tokens", 0)
                if isinstance(usage, dict)
                else getattr(usage, "input_tokens", 0)
            )
            completion_tokens = (
                usage.get("output_tokens", 0)
                if isinstance(usage, dict)
                else getattr(usage, "output_tokens", 0)
            )

            return LLMResponse(
                content=response.content,
                model=request.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        except KairosError:
            raise
        except Exception as e:
            raise KairosError(
                f"OpenAI completion failed: {e}",
                code="LLM_CALL_FAILED",
            ) from e
