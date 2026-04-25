"""
Policy enforcement nodes for agent orchestration.

Pre-tool policy: Checks if tools are allowed before execution
Post-tool policy: Sanitizes tool outputs for PII
"""

from langchain_core.messages import AIMessage

from ai_platform.agents.state import AgentState
from ai_platform.policy import PolicyEngine, PIIDetector
from ai_platform.core.observability import trace_operation


def pre_tool_policy_node(
    state: AgentState,
    policy_engine: PolicyEngine
) -> AgentState:
    """
    Pre-tool policy check node.
    
    Checks if requested tools are allowed in the current environment
    based on policy YAML configuration.
    
    Args:
        state: Current agent state
        policy_engine: Policy engine instance
        
    Returns:
        Updated state with policy_results populated
    """
    environment = state["environment"]
    messages = state["messages"]
    
    # Get the last AI message with tool calls
    policy_results = list(state["policy_results"])
    
    if not messages:
        return {**state, "policy_results": policy_results}
    
    last_message = messages[-1]
    
    # Check if last message has tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Trace policy evaluation
        with trace_operation(
            "policy.pre_check",
            attributes={
                "policy.environment": environment,
                "policy.tool_count": len(last_message.tool_calls),
            }
        ) as span:
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                
                # Evaluate tool permission
                result = policy_engine.evaluate(tool_name, environment)
                policy_results.append(result)
            
            # Record policy decisions
            allowed_count = sum(1 for r in policy_results if r.allowed)
            denied_count = len(policy_results) - allowed_count
            span.set_attribute("policy.allowed_count", allowed_count)
            span.set_attribute("policy.denied_count", denied_count)
    
    return {
        **state,
        "policy_results": policy_results
    }


def post_tool_policy_node(
    state: AgentState,
    policy_engine: PolicyEngine,
    pii_detector: PIIDetector
) -> AgentState:
    """
    Post-tool policy check node.
    
    Sanitizes tool outputs for PII based on environment policy configuration.
    
    Args:
        state: Current agent state
        policy_engine: Policy engine instance
        pii_detector: PII detector instance
        
    Returns:
        Updated state with sanitized tool results
    """
    environment = state["environment"]
    
    # Check if PII enforcement is enabled for this environment
    pii_config = policy_engine.get_pii_config(environment)
    
    if not pii_config.get("enabled", False):
        # PII enforcement disabled, pass through
        return state
    
    # Trace PII sanitization
    with trace_operation(
        "policy.pii_sanitize",
        attributes={
            "policy.environment": environment,
            "pii.enabled": True,
        }
    ) as span:
        # Sanitize tool results
        tool_results = state["tool_results"]
        sanitized_results = []
        pii_found_count = 0
        
        for result in tool_results:
            # Sanitize the output if it contains PII
            if result.output and isinstance(result.output, (str, dict)):
                if isinstance(result.output, str):
                    original = result.output
                    sanitized_output = pii_detector.sanitize(result.output)
                    if original != sanitized_output:
                        pii_found_count += 1
                else:
                    sanitized_output = pii_detector.sanitize_dict(result.output)
                
                # Create a new result with sanitized output
                sanitized_result = result.model_copy(update={"output": sanitized_output})
                sanitized_results.append(sanitized_result)
            else:
                sanitized_results.append(result)
        
        span.set_attribute("pii.instances_found", pii_found_count)
        span.set_attribute("pii.results_processed", len(tool_results))
    
    return {
        **state,
        "tool_results": sanitized_results
    }

