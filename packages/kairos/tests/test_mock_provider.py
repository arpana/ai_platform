from __future__ import annotations

import pytest

from ai_platform.core.models import LLMRequest, LLMResponse
from ai_platform.kairos.providers.mock import MockProvider


@pytest.mark.asyncio
async def test_mock_returns_llm_response():
    provider = MockProvider()
    request = LLMRequest(prompt="Hello world")

    result = await provider.complete(request)

    assert isinstance(result, LLMResponse)
    assert result.content.startswith("[MOCK]")
    assert result.model == request.model
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0


@pytest.mark.asyncio
async def test_mock_echoes_prompt():
    provider = MockProvider()
    request = LLMRequest(prompt="Tell me about banking")

    result = await provider.complete(request)

    assert "Tell me about banking" in result.content


@pytest.mark.asyncio
async def test_mock_respects_model_field():
    provider = MockProvider()
    request = LLMRequest(prompt="Hi", model="custom-model-v1")

    result = await provider.complete(request)

    assert result.model == "custom-model-v1"
