import time
from typing import Any

from ai_platform.core.models import ToolResult
from ai_platform.tools.base import BaseTool


class RecommendationEngineTool(BaseTool):
    """Get personalized product recommendations for a customer."""
    
    # Mock recommendations database
    BANKING_RECOMMENDATIONS = {
        "savings": [
            {"name": "High-Yield Savings Account", "apy": "4.5%", "min_balance": 1000},
            {"name": "Premium Savings Account", "apy": "5.0%", "min_balance": 10000}
        ],
        "credit": [
            {"name": "Rewards Credit Card", "apr": "15.99%", "rewards": "2% cashback"},
            {"name": "Travel Credit Card", "apr": "17.99%", "rewards": "3x points on travel"}
        ],
        "investment": [
            {"name": "Index Fund Portfolio", "risk": "low", "expected_return": "7-9%"},
            {"name": "Growth Stock Portfolio", "risk": "high", "expected_return": "12-15%"}
        ]
    }
    
    RETAIL_RECOMMENDATIONS = {
        "electronics": [
            {"name": "Wireless Earbuds Pro", "price": 249.99, "rating": 4.8},
            {"name": "Smart Watch Ultra", "price": 399.99, "rating": 4.7},
            {"name": "4K Laptop Display", "price": 599.99, "rating": 4.9}
        ],
        "clothing": [
            {"name": "Premium Cotton T-Shirt", "price": 29.99, "rating": 4.6},
            {"name": "Designer Jeans", "price": 89.99, "rating": 4.7},
            {"name": "Winter Jacket", "price": 149.99, "rating": 4.8}
        ],
        "home": [
            {"name": "Smart Speaker", "price": 99.99, "rating": 4.5},
            {"name": "Robot Vacuum", "price": 299.99, "rating": 4.6},
            {"name": "Air Purifier", "price": 199.99, "rating": 4.7}
        ]
    }
    
    @property
    def name(self) -> str:
        return "recommendation_engine"
    
    @property
    def description(self) -> str:
        return "Get personalized product recommendations for a customer"
    
    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Unique identifier for the customer"
                },
                "category": {
                    "type": "string",
                    "description": "Optional category to filter recommendations (e.g., 'savings', 'electronics')"
                },
                "environment": {
                    "type": "string",
                    "description": "Environment context (banking or retail) - auto-detected if not provided"
                }
            },
            "required": ["customer_id"]
        }
    
    @property
    def environment_scopes(self) -> list[str]:
        return ["banking", "retail"]
    
    async def execute(self, input_data: dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        # Validate required fields
        if "customer_id" not in input_data:
            return ToolResult(
                tool_name=self.name,
                output=None,
                error="Missing required field: customer_id",
                blocked=False,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        customer_id = input_data["customer_id"]
        category = input_data.get("category")
        environment = input_data.get("environment", "banking")  # Default to banking
        
        # Determine which recommendations to use based on environment
        if environment == "banking":
            recommendations_db = self.BANKING_RECOMMENDATIONS
            default_categories = ["savings", "credit", "investment"]
        else:  # retail
            recommendations_db = self.RETAIL_RECOMMENDATIONS
            default_categories = ["electronics", "clothing", "home"]
        
        # Filter by category if specified
        if category:
            if category not in recommendations_db:
                return ToolResult(
                    tool_name=self.name,
                    output=None,
                    error=f"Unknown category: {category}. Available categories: {', '.join(recommendations_db.keys())}",
                    blocked=False,
                    latency_ms=(time.time() - start_time) * 1000
                )
            recommendations = {category: recommendations_db[category]}
        else:
            # Return all categories
            recommendations = recommendations_db
        
        result = {
            "customer_id": customer_id,
            "environment": environment,
            "recommendations": recommendations,
            "total_items": sum(len(items) for items in recommendations.values())
        }
        
        return ToolResult(
            tool_name=self.name,
            output=result,
            error=None,
            blocked=False,
            latency_ms=(time.time() - start_time) * 1000
        )
