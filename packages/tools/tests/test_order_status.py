import pytest
from ai_platform.tools import OrderStatusTool


@pytest.fixture
def order_status():
    """Create an OrderStatusTool instance."""
    return OrderStatusTool()


def test_order_status_properties(order_status):
    """Test OrderStatusTool basic properties."""
    assert order_status.name == "order_status"
    assert order_status.description == "Look up the current status of a customer order"
    assert order_status.environment_scopes == ["retail"]
    assert "order_id" in order_status.input_schema["properties"]


@pytest.mark.asyncio
async def test_order_status_shipped(order_status):
    """Test looking up a shipped order."""
    result = await order_status.execute({"order_id": "ORD-001"})
    
    assert result.error is None
    assert result.output["status"] == "shipped"
    assert result.output["order_id"] == "ORD-001"
    assert result.output["tracking_number"] == "1Z999AA10123456784"
    assert "Laptop" in result.output["items"]
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_order_status_processing(order_status):
    """Test looking up a processing order."""
    result = await order_status.execute({"order_id": "ORD-002"})
    
    assert result.error is None
    assert result.output["status"] == "processing"
    assert result.output["tracking_number"] is None
    assert result.output["estimated_delivery"] == "2026-04-08"


@pytest.mark.asyncio
async def test_order_status_delivered(order_status):
    """Test looking up a delivered order."""
    result = await order_status.execute({"order_id": "ORD-003"})
    
    assert result.error is None
    assert result.output["status"] == "delivered"
    assert result.output["estimated_delivery"] == "2026-04-03"


@pytest.mark.asyncio
async def test_order_status_cancelled(order_status):
    """Test looking up a cancelled order."""
    result = await order_status.execute({"order_id": "ORD-004"})
    
    assert result.error is None
    assert result.output["status"] == "cancelled"
    assert result.output["tracking_number"] is None
    assert result.output["estimated_delivery"] is None


@pytest.mark.asyncio
async def test_order_status_not_found(order_status):
    """Test error handling for non-existent order."""
    result = await order_status.execute({"order_id": "ORD-999"})
    
    assert result.error is not None
    assert "not found" in result.error.lower()
    assert "ORD-999" in result.error


@pytest.mark.asyncio
async def test_order_status_missing_order_id(order_status):
    """Test error handling for missing order_id."""
    result = await order_status.execute({})
    
    assert result.error is not None
    assert "Missing required field" in result.error
    assert "order_id" in result.error


@pytest.mark.asyncio
async def test_order_status_all_orders_have_required_fields(order_status):
    """Test that all mock orders have required fields."""
    valid_order_ids = ["ORD-001", "ORD-002", "ORD-003", "ORD-004"]
    
    for order_id in valid_order_ids:
        result = await order_status.execute({"order_id": order_id})
        assert result.error is None
        assert "order_id" in result.output
        assert "status" in result.output
        assert "items" in result.output
        assert "total" in result.output
