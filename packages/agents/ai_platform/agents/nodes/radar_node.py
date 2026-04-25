"""
Radar Check Node - Tech radar governance enforcement.

Validates tool calls against tech radar before execution:
- Blocks tools with STOP status
- Warns for tools under review
- Allows approved tools
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_platform.radar import TechRadar

from ai_platform.agents.state import AgentState
from ai_platform.core.models import RadarStatus
from ai_platform.core.exceptions import RadarBlockedError


logger = logging.getLogger(__name__)


def radar_check_node(state: AgentState, radar_registry: "TechRadar") -> AgentState:
    """
    Check tool calls against tech radar before execution.
    
    Processes the last AI message and:
    - Blocks tools with STOP status (raises RadarBlockedError)
    - Warns for tools UNDER_REVIEW
    - Allows APPROVED tools
    
    Args:
        state: Current agent state with messages
        radar_registry: Tech radar registry
        
    Returns:
        Updated state (unchanged if all tools allowed)
        
    Raises:
        RadarBlockedError: If any tool has STOP status
    """
    messages = state.get("messages", [])
    
    if not messages:
        return state
    
    last_message = messages[-1]
    
    # Check if last message is an AI message with tool calls
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        # No tool calls to check
        return state
    
    # Check each tool call against radar
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        
        if not tool_name:
            continue
        
        # Get radar status and enforce
        entry = radar_registry.check_and_enforce(tool_name)
        
        # Log warnings for under_review tools
        if entry.status == RadarStatus.UNDER_REVIEW:
            logger.warning(
                f"Tool '{tool_name}' is under review - {entry.notes}. "
                f"Allowing execution but this tool may be deprecated soon."
            )
    
    # All tools passed - return unchanged state
    return state
