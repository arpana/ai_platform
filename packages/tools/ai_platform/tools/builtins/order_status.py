import time
from typing import Any

from ai_platform.core.models import ToolResult
from ai_platform.tools.base import BaseTool


class OrderStatusTool(BaseTool):
    """Look up the current status of a customer order."""
    
    # Mock order database
    MOCK_ORDERS = {
        "ORD-001": {
            "order_id": "ORD-001",
            "status": "shipped",
            "items": ["Laptop", "Mouse"],
            "total": 1299.99,
            "tracking_number": "1Z999AA10123456784",
            "estimated_delivery": "2026-04-06"
        },
        "ORD-002": {
            "order_id": "ORD-002",
            "status": "processing",
            "items": ["Headphones"],
            "total": 199.99,
            "tracking_number": None,
            "estimated_delivery": "2026-04-08"
        },
        "ORD-003": {
            "order_id": "ORD-003",
            "status": "delivered",
            "items": ["Keyboard", "Monitor"],
            "total": 549.99,
            "tracking_number": "1Z999AA10123456785",
            "estimated_delivery": "2026-04-03"
        },
        "ORD-004": {
            "order_id": "ORD-004",
            "status": "cancelled",
            "items": ["Tablet"],
            "total": 499.99,
            "tracking_number": None,
            "estimated_delivery": None
        }
    }
    
    @property
    def name(self) -> str:
        return "order_status"
    
    @property
    def description(self) -> str:
        return "Look up the current status of a customer order"
    
    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique order identifier"
                }
            },
            "required": ["order_id"]
        }
    
    @property
    def environment_scopes(self) -> list[str]:
        return ["retail"]
    
    async def execute(self, input_data: dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        # Validate required fields
        if "order_id" not in input_data:
            return ToolResult(
                tool_name=self.name,
                output=None,
                error="Missing required field: order_id",
                blocked=False,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        order_id = input_data["order_id"]
        
        # Look up order in mock database
        if order_id not in self.MOCK_ORDERS:
            return ToolResult(
                tool_name=self.name,
                output=None,
                error=f"Order not found: {order_id}",
                blocked=False,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        order_info = self.MOCK_ORDERS[order_id]
        
        return ToolResult(
            tool_name=self.name,
            output=order_info,
            error=None,
            blocked=False,
            latency_ms=(time.time() - start_time) * 1000
        )
