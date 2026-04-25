import time
from typing import Any

from ai_platform.core.models import ToolResult
from ai_platform.tools.base import BaseTool


class LoanCheckerTool(BaseTool):
    """Check loan eligibility based on customer income and requested amount."""
    
    @property
    def name(self) -> str:
        return "loan_checker"
    
    @property
    def description(self) -> str:
        return "Check loan eligibility based on customer income and requested amount"
    
    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Unique identifier for the customer"
                },
                "annual_income": {
                    "type": "number",
                    "description": "Customer's annual income in USD"
                },
                "loan_amount": {
                    "type": "number",
                    "description": "Requested loan amount in USD"
                }
            },
            "required": ["customer_id", "annual_income", "loan_amount"]
        }
    
    @property
    def environment_scopes(self) -> list[str]:
        return ["banking"]
    
    async def execute(self, input_data: dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        # Validate required fields
        required_fields = ["customer_id", "annual_income", "loan_amount"]
        missing_fields = [f for f in required_fields if f not in input_data]
        if missing_fields:
            return ToolResult(
                tool_name=self.name,
                output=None,
                error=f"Missing required fields: {', '.join(missing_fields)}",
                blocked=False,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        customer_id = input_data["customer_id"]
        annual_income = input_data["annual_income"]
        loan_amount = input_data["loan_amount"]
        
        # Validate numeric values
        try:
            annual_income = float(annual_income)
            loan_amount = float(loan_amount)
        except (ValueError, TypeError):
            return ToolResult(
                tool_name=self.name,
                output=None,
                error="annual_income and loan_amount must be numeric values",
                blocked=False,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        # Mock eligibility logic: loan_amount <= 5 * annual_income
        max_approved_amount = annual_income * 5
        eligible = loan_amount <= max_approved_amount
        
        # Determine interest rate tier based on income
        if annual_income >= 100000:
            interest_rate = "3.5%"
            tier = "premium"
        elif annual_income >= 50000:
            interest_rate = "4.5%"
            tier = "standard"
        else:
            interest_rate = "5.5%"
            tier = "basic"
        
        result = {
            "customer_id": customer_id,
            "eligible": eligible,
            "requested_amount": loan_amount,
            "max_approved_amount": max_approved_amount,
            "interest_rate": interest_rate,
            "tier": tier,
            "reason": "Eligible" if eligible else f"Requested amount exceeds maximum ({max_approved_amount})"
        }
        
        return ToolResult(
            tool_name=self.name,
            output=result,
            error=None,
            blocked=False,
            latency_ms=(time.time() - start_time) * 1000
        )
