import pytest
from ai_platform.tools import LoanCheckerTool


@pytest.fixture
def loan_checker():
    """Create a LoanCheckerTool instance."""
    return LoanCheckerTool()


def test_loan_checker_properties(loan_checker):
    """Test LoanCheckerTool basic properties."""
    assert loan_checker.name == "loan_checker"
    assert loan_checker.description == "Check loan eligibility based on customer income and requested amount"
    assert loan_checker.environment_scopes == ["banking"]
    assert "customer_id" in loan_checker.input_schema["properties"]
    assert "annual_income" in loan_checker.input_schema["properties"]
    assert "loan_amount" in loan_checker.input_schema["properties"]


@pytest.mark.asyncio
async def test_loan_checker_eligible(loan_checker):
    """Test loan approval for eligible customer."""
    result = await loan_checker.execute({
        "customer_id": "CUST-001",
        "annual_income": 80000,
        "loan_amount": 200000
    })
    
    assert result.error is None
    assert result.output["eligible"] is True
    assert result.output["customer_id"] == "CUST-001"
    assert result.output["max_approved_amount"] == 400000.0
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_loan_checker_not_eligible(loan_checker):
    """Test loan rejection for ineligible customer."""
    result = await loan_checker.execute({
        "customer_id": "CUST-002",
        "annual_income": 50000,
        "loan_amount": 300000
    })
    
    assert result.error is None
    assert result.output["eligible"] is False
    assert result.output["max_approved_amount"] == 250000.0
    assert "exceeds maximum" in result.output["reason"]


@pytest.mark.asyncio
async def test_loan_checker_high_income_tier(loan_checker):
    """Test premium tier for high-income customers."""
    result = await loan_checker.execute({
        "customer_id": "CUST-003",
        "annual_income": 150000,
        "loan_amount": 500000
    })
    
    assert result.error is None
    assert result.output["tier"] == "premium"
    assert result.output["interest_rate"] == "3.5%"


@pytest.mark.asyncio
async def test_loan_checker_medium_income_tier(loan_checker):
    """Test standard tier for medium-income customers."""
    result = await loan_checker.execute({
        "customer_id": "CUST-004",
        "annual_income": 60000,
        "loan_amount": 100000
    })
    
    assert result.error is None
    assert result.output["tier"] == "standard"
    assert result.output["interest_rate"] == "4.5%"


@pytest.mark.asyncio
async def test_loan_checker_low_income_tier(loan_checker):
    """Test basic tier for lower-income customers."""
    result = await loan_checker.execute({
        "customer_id": "CUST-005",
        "annual_income": 35000,
        "loan_amount": 50000
    })
    
    assert result.error is None
    assert result.output["tier"] == "basic"
    assert result.output["interest_rate"] == "5.5%"


@pytest.mark.asyncio
async def test_loan_checker_missing_field(loan_checker):
    """Test error handling for missing required field."""
    result = await loan_checker.execute({
        "customer_id": "CUST-006",
        "annual_income": 80000
        # Missing loan_amount
    })
    
    assert result.error is not None
    assert "Missing required fields" in result.error
    assert "loan_amount" in result.error


@pytest.mark.asyncio
async def test_loan_checker_invalid_income_type(loan_checker):
    """Test error handling for invalid data type."""
    result = await loan_checker.execute({
        "customer_id": "CUST-007",
        "annual_income": "not_a_number",
        "loan_amount": 100000
    })
    
    assert result.error is not None
    assert "numeric" in result.error.lower()


@pytest.mark.asyncio
async def test_loan_checker_edge_case_exact_max(loan_checker):
    """Test exact maximum loan amount."""
    result = await loan_checker.execute({
        "customer_id": "CUST-008",
        "annual_income": 100000,
        "loan_amount": 500000  # Exactly 5x income
    })
    
    assert result.error is None
    assert result.output["eligible"] is True
    assert result.output["requested_amount"] == 500000.0
