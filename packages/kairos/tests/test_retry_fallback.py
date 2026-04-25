from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_platform.core.exceptions import KairosError
from ai_platform.core.models import LLMRequest, LLMResponse
from ai_platform.kairos.providers.mock import MockProvider
from ai_platform.kairos.retry_fallback import FallbackProvider, retry_with_backoff


@pytest.mark.asyncio
async def test_retry_succeeds_on_first_attempt():
    call_count = 0

    async def succeed():
        nonlocal call_count
        call_count += 1
        return LLMResponse(content="ok", model="m")

    result = await retry_with_backoff(succeed, max_attempts=3, initial_delay=0.01)

    assert result.content == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_retries_on_kairos_error():
    call_count = 0

    async def fail_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise KairosError("transient")
        return LLMResponse(content="recovered", model="m")

    result = await retry_with_backoff(
        fail_then_succeed,
        max_attempts=3,
        initial_delay=0.01,
    )

    assert result.content == "recovered"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_raises_after_max_attempts():
    async def always_fail():
        raise KairosError("permanent")

    with pytest.raises(KairosError, match="permanent"):
        await retry_with_backoff(
            always_fail,
            max_attempts=2,
            initial_delay=0.01,
        )


@pytest.mark.asyncio
async def test_fallback_uses_second_provider():
    failing_provider = AsyncMock(spec=MockProvider)
    failing_provider.complete = AsyncMock(side_effect=KairosError("down"))

    good_provider = MockProvider()

    fallback = FallbackProvider([failing_provider, good_provider])
    request = LLMRequest(prompt="test fallback")

    result = await fallback.complete(request)

    assert isinstance(result, LLMResponse)
    assert "[MOCK]" in result.content


@pytest.mark.asyncio
async def test_fallback_raises_when_all_fail():
    p1 = AsyncMock(spec=MockProvider)
    p1.complete = AsyncMock(side_effect=KairosError("fail-1"))

    p2 = AsyncMock(spec=MockProvider)
    p2.complete = AsyncMock(side_effect=KairosError("fail-2"))

    fallback = FallbackProvider([p1, p2])
    request = LLMRequest(prompt="all fail")

    with pytest.raises(KairosError, match="All 2 providers failed"):
        await fallback.complete(request)
