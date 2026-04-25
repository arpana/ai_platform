from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_returns_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ready_returns_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "environment" in data
    assert "tools_registered" in data
    assert "kairos_provider" in data


def test_ready_reflects_settings(client, monkeypatch):
    monkeypatch.setenv("AIP_KAIROS_PROVIDER", "openai")
    from ai_platform.core.config import get_settings

    get_settings.cache_clear()

    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["kairos_provider"] == "openai"
