from functools import partial
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from ai_platform.agents.state import AgentState
from ai_platform.agents.nodes.rag_node import rag_node
from ai_platform.agents.nodes.policy_node import pre_tool_policy_node, post_tool_policy_node
from ai_platform.agents.nodes.radar_node import radar_check_node
from ai_platform.tools import ToolRegistry, wrap_tools_for_environment
from ai_platform.core.models import ToolResult
from ai_platform.core.observability import trace_operation
from ai_platform.rag.retriever import RAGRetriever
from ai_platform.policy import PolicyEngine, PIIDetector
from ai_platform.radar import TechRadar


# Maximum iterations to prevent infinite loops
MAX_ITERATIONS = 15


async def reasoning_node(state: AgentState, llm: ChatOpenAI, registry: ToolRegistry) -> AgentState:
    """
    LLM reasoning node - decides next action.

    Calls the LLM with current messages and available tools.
    If the LLM wants to call tools, appends an AIMessage with tool_calls.
    Otherwise, appends the final response AIMessage.

    Args:
        state: Current agent state
        llm: ChatOpenAI instance (from Kairos)
        registry: Tool registry for getting environment-scoped tools

    Returns:
        Updated state with new message and incremented iteration count
    """
    messages = state["messages"]
    environment = state["environment"]

    # Get tools for this environment and wrap for LangChain
    langchain_tools = wrap_tools_for_environment(registry, environment)

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(langchain_tools)

    # Call LLM asynchronously (avoids blocking the event loop in FastAPI)
    response = await llm_with_tools.ainvoke(messages)

    # Append AI response to messages
    updated_messages = messages + [response]

    # Increment iteration count
    updated_iteration_count = state["iteration_count"] + 1

    return {**state, "messages": updated_messages, "iteration_count": updated_iteration_count}


async def tool_node(state: AgentState, registry: ToolRegistry) -> AgentState:
    """
    Tool execution node.

    Executes tools requested by the LLM in the last message.
    Appends ToolMessage results and updates tool_results list.

    Args:
        state: Current agent state
        registry: Tool registry for executing tools

    Returns:
        Updated state with tool messages and results
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Extract tool calls from the last AI message
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        # Safety check - should not happen if routing is correct
        return state

    tool_messages = []
    tool_results = list(state["tool_results"])

    # Execute each tool
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        tool_call_id = tool_call["id"]

        # Trace tool execution
        with trace_operation(
            f"tool.{tool_name}",
            attributes={
                "tool.name": tool_name,
                "tool.input": str(tool_input)[:500],  # Truncate for brevity
            }
        ):
            try:
                # Get the tool from registry
                tool = registry.get(tool_name)

                # Execute the tool (async)
                result = await tool.execute(tool_input)

                # Create ToolMessage with the result
                if result.error:
                    tool_message = ToolMessage(
                        content=f"Error: {result.error}", tool_call_id=tool_call_id, name=tool_name
                    )
                else:
                    tool_message = ToolMessage(
                        content=str(result.output), tool_call_id=tool_call_id, name=tool_name
                    )

                tool_messages.append(tool_message)
                tool_results.append(result)

            except Exception as e:
                # Handle tool execution errors
                error_message = ToolMessage(
                    content=f"Error executing tool: {str(e)}", tool_call_id=tool_call_id, name=tool_name
                )
                tool_messages.append(error_message)

                # Create error ToolResult
                error_result = ToolResult(
                    tool_name=tool_name, output=None, error=str(e), blocked=False, latency_ms=0.0
                )
                tool_results.append(error_result)

    # Append tool messages to conversation
    updated_messages = messages + tool_messages

    return {**state, "messages": updated_messages, "tool_results": tool_results}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Routing function that decides whether to continue with tools or end.

    Continues if:
    1. Last message has tool_calls (LLM wants to use tools)
    2. Iteration count is below the maximum

    Args:
        state: Current agent state

    Returns:
        "tools" to execute tools and continue, or "end" to finish
    """
    messages = state["messages"]
    iteration_count = state["iteration_count"]

    # Safety check: max iterations
    if iteration_count >= MAX_ITERATIONS:
        return "end"

    # Check if last message has tool calls
    if not messages:
        return "end"

    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "end"


def create_agent_graph(
    llm: ChatOpenAI,
    registry: ToolRegistry,
    policy_engine: PolicyEngine,
    pii_detector: PIIDetector,
    radar_registry: TechRadar,
    retriever: RAGRetriever | None = None,
) -> StateGraph:
    """
    Create the LangGraph agent workflow.
    
    Graph flow:
    START → rag → reasoning → should_continue?
                               ├── "tools" → pre_policy → radar_check → tools → post_policy → reasoning (loop)
                               └── "end"  → END
    
    Args:
        llm: ChatOpenAI instance for reasoning
        registry: ToolRegistry with registered tools
        policy_engine: PolicyEngine for tool governance
        pii_detector: PIIDetector for output sanitization
        radar_registry: TechRadar for technology governance
        retriever: Optional RAGRetriever for document retrieval
        
    Returns:
        Compiled StateGraph ready for execution
    """
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("rag", partial(rag_node, retriever=retriever))
    workflow.add_node("reasoning", partial(reasoning_node, llm=llm, registry=registry))
    workflow.add_node("pre_policy", partial(pre_tool_policy_node, policy_engine=policy_engine))
    workflow.add_node("radar_check", partial(radar_check_node, radar_registry=radar_registry))
    workflow.add_node("tools", partial(tool_node, registry=registry))
    workflow.add_node("post_policy", partial(post_tool_policy_node, policy_engine=policy_engine, pii_detector=pii_detector))

    # Define the flow
    workflow.set_entry_point("rag")
    workflow.add_edge("rag", "reasoning")
    workflow.add_conditional_edges("reasoning", should_continue, {"tools": "pre_policy", "end": END})
    workflow.add_edge("pre_policy", "radar_check")
    workflow.add_edge("radar_check", "tools")
    workflow.add_edge("tools", "post_policy")
    workflow.add_edge("post_policy", "reasoning")

    return workflow.compile()
