from __future__ import annotations

from abc import ABC, abstractmethod

from ai_platform.core.models import LLMRequest, LLMResponse


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers in the Kairos gateway."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion for the given request."""
