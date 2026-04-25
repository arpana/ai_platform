from __future__ import annotations

import pytest

from ai_platform.core.exceptions import KairosError
from ai_platform.kairos.providers.openai import OpenAIProvider


def test_openai_requires_api_key():
    with pytest.raises(KairosError, match="API key is required"):
        OpenAIProvider(api_key="")


def test_openai_raises_kairos_error_for_missing_key():
    with pytest.raises(KairosError) as exc_info:
        OpenAIProvider(api_key="")

    assert exc_info.value.code == "MISSING_API_KEY"


def test_openai_accepts_valid_key():
    provider = OpenAIProvider(api_key="sk-test-key-12345")
    assert provider._api_key == "sk-test-key-12345"
