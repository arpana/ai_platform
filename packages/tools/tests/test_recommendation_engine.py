import pytest
from ai_platform.tools import RecommendationEngineTool


@pytest.fixture
def recommendation_engine():
    """Create a RecommendationEngineTool instance."""
    return RecommendationEngineTool()


def test_recommendation_engine_properties(recommendation_engine):
    """Test RecommendationEngineTool basic properties."""
    assert recommendation_engine.name == "recommendation_engine"
    assert recommendation_engine.description == "Get personalized product recommendations for a customer"
    assert set(recommendation_engine.environment_scopes) == {"banking", "retail"}
    assert "customer_id" in recommendation_engine.input_schema["properties"]
    assert "category" in recommendation_engine.input_schema["properties"]


@pytest.mark.asyncio
async def test_recommendation_engine_banking(recommendation_engine):
    """Test banking recommendations without category filter."""
    result = await recommendation_engine.execute({
        "customer_id": "CUST-001",
        "environment": "banking"
    })
    
    assert result.error is None
    assert result.output["environment"] == "banking"
    assert "savings" in result.output["recommendations"]
    assert "credit" in result.output["recommendations"]
    assert "investment" in result.output["recommendations"]
    assert result.output["total_items"] > 0
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_recommendation_engine_banking_savings_category(recommendation_engine):
    """Test banking recommendations with savings category."""
    result = await recommendation_engine.execute({
        "customer_id": "CUST-002",
        "environment": "banking",
        "category": "savings"
    })
    
    assert result.error is None
    assert "savings" in result.output["recommendations"]
    assert "credit" not in result.output["recommendations"]
    assert len(result.output["recommendations"]["savings"]) > 0


@pytest.mark.asyncio
async def test_recommendation_engine_banking_credit_category(recommendation_engine):
    """Test banking recommendations with credit category."""
    result = await recommendation_engine.execute({
        "customer_id": "CUST-003",
        "environment": "banking",
        "category": "credit"
    })
    
    assert result.error is None
    assert "credit" in result.output["recommendations"]
    assert "savings" not in result.output["recommendations"]


@pytest.mark.asyncio
async def test_recommendation_engine_retail(recommendation_engine):
    """Test retail recommendations without category filter."""
    result = await recommendation_engine.execute({
        "customer_id": "CUST-004",
        "environment": "retail"
    })
    
    assert result.error is None
    assert result.output["environment"] == "retail"
    assert "electronics" in result.output["recommendations"]
    assert "clothing" in result.output["recommendations"]
    assert "home" in result.output["recommendations"]
    assert result.output["total_items"] > 0


@pytest.mark.asyncio
async def test_recommendation_engine_retail_electronics_category(recommendation_engine):
    """Test retail recommendations with electronics category."""
    result = await recommendation_engine.execute({
        "customer_id": "CUST-005",
        "environment": "retail",
        "category": "electronics"
    })
    
    assert result.error is None
    assert "electronics" in result.output["recommendations"]
    assert "clothing" not in result.output["recommendations"]
    electronics = result.output["recommendations"]["electronics"]
    assert len(electronics) > 0
    # Verify structure of electronics recommendations
    assert "name" in electronics[0]
    assert "price" in electronics[0]
    assert "rating" in electronics[0]


@pytest.mark.asyncio
async def test_recommendation_engine_retail_clothing_category(recommendation_engine):
    """Test retail recommendations with clothing category."""
    result = await recommendation_engine.execute({
        "customer_id": "CUST-006",
        "environment": "retail",
        "category": "clothing"
    })
    
    assert result.error is None
    assert "clothing" in result.output["recommendations"]
    assert len(result.output["recommendations"]["clothing"]) > 0


@pytest.mark.asyncio
async def test_recommendation_engine_invalid_category(recommendation_engine):
    """Test error handling for invalid category."""
    result = await recommendation_engine.execute({
        "customer_id": "CUST-007",
        "environment": "banking",
        "category": "invalid_category"
    })
    
    assert result.error is not None
    assert "Unknown category" in result.error
    assert "invalid_category" in result.error


@pytest.mark.asyncio
async def test_recommendation_engine_missing_customer_id(recommendation_engine):
    """Test error handling for missing customer_id."""
    result = await recommendation_engine.execute({
        "environment": "banking"
    })
    
    assert result.error is not None
    assert "Missing required field" in result.error
    assert "customer_id" in result.error


@pytest.mark.asyncio
async def test_recommendation_engine_default_environment(recommendation_engine):
    """Test that default environment is banking when not specified."""
    result = await recommendation_engine.execute({
        "customer_id": "CUST-008"
    })
    
    assert result.error is None
    assert result.output["environment"] == "banking"
    assert "savings" in result.output["recommendations"]


@pytest.mark.asyncio
async def test_recommendation_engine_banking_has_different_data_than_retail(recommendation_engine):
    """Test that banking and retail return different recommendation types."""
    banking_result = await recommendation_engine.execute({
        "customer_id": "CUST-009",
        "environment": "banking"
    })
    
    retail_result = await recommendation_engine.execute({
        "customer_id": "CUST-010",
        "environment": "retail"
    })
    
    assert banking_result.error is None
    assert retail_result.error is None
    
    # Banking should have financial products
    banking_categories = set(banking_result.output["recommendations"].keys())
    assert "savings" in banking_categories or "credit" in banking_categories
    
    # Retail should have retail products
    retail_categories = set(retail_result.output["recommendations"].keys())
    assert "electronics" in retail_categories or "clothing" in retail_categories
    
    # They shouldn't overlap
    assert banking_categories != retail_categories
