from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_all_radar_status(client):
    response = client.get("/radar/status")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 6
    for entry in data:
        assert "name" in entry
        assert "status" in entry
        assert "category" in entry


def test_get_approved_tool_status(client):
    response = client.get("/radar/status/loan_checker")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "loan_checker"
    assert data["status"] == "approved"


def test_get_under_review_tool_status(client):
    response = client.get("/radar/status/product_search")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "product_search"
    assert data["status"] == "under_review"


def test_get_stopped_tool_status(client):
    response = client.get("/radar/status/external_pricing_api")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "external_pricing_api"
    assert data["status"] == "stop"


def test_get_unknown_tool_status(client):
    response = client.get("/radar/status/nonexistent_tool")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "nonexistent_tool"
    assert data["status"] == "stop"
