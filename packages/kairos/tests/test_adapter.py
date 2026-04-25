from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_platform.core.config import Settings
from ai_platform.core.exceptions import KairosError
from ai_platform.core.models import LLMRequest, LLMResponse
from ai_platform.kairos.adapter import KairosAdapter
from ai_platform.kairos.providers.mock import MockProvider
from ai_platform.kairos.providers.openai import OpenAIProvider
from ai_platform.kairos.retry_fallback import FallbackProvider


def test_adapter_creates_mock_provider():
    settings = Settings(kairos_provider="mock")
    adapter = KairosAdapter(settings=settings)

    assert isinstance(adapter.provider, MockProvider)


def test_adapter_creates_openai_provider():
    settings = Settings(kairos_provider="openai", kairos_api_key="sk-test")
    adapter = KairosAdapter(settings=settings)

    assert isinstance(adapter.provider, OpenAIProvider)


def test_adapter_creates_fallback_provider():
    settings = Settings(kairos_provider="fallback", kairos_api_key="sk-test")
    adapter = KairosAdapter(settings=settings)

    assert isinstance(adapter.provider, FallbackProvider)


def test_adapter_raises_on_unknown_provider():
    settings = Settings(kairos_provider="nonexistent")

    with pytest.raises(KairosError, match="Unknown kairos provider"):
        KairosAdapter(settings=settings)


@pytest.mark.asyncio
async def test_adapter_complete_delegates_to_provider():
    settings = Settings(kairos_provider="mock")
    adapter = KairosAdapter(settings=settings)
    request = LLMRequest(prompt="Hello adapter")

    result = await adapter.complete(request)

    assert isinstance(result, LLMResponse)
    assert "[MOCK]" in result.content


@pytest.mark.asyncio
async def test_adapter_complete_for_environment_overrides_model():
    settings = Settings(kairos_provider="mock")
    adapter = KairosAdapter(settings=settings)
    request = LLMRequest(prompt="banking query", model="original-model")

    result = await adapter.complete_for_environment(request, "banking")

    assert isinstance(result, LLMResponse)
