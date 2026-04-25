from typing import Any, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from .base import BaseTool
from .registry import ToolRegistry


def _json_schema_to_pydantic(
    schema: dict[str, Any], model_name: str = "ToolInput"
) -> Type[BaseModel]:
    """
    Convert a JSON Schema dict to a Pydantic model dynamically.

    Args:
        schema: JSON Schema dictionary with 'properties' and 'required' keys
        model_name: Name for the generated Pydantic model

    Returns:
        A dynamically created Pydantic BaseModel class
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Map JSON Schema types to Python types
    type_mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    # Build field definitions for create_model
    field_definitions = {}
    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        python_type = type_mapping.get(json_type, str)
        description = field_schema.get("description", "")

        # Determine if field is required or optional
        if field_name in required:
            # Required field
            field_definitions[field_name] = (python_type, Field(description=description))
        else:
            # Optional field with default None
            field_definitions[field_name] = (
                python_type | None,
                Field(default=None, description=description),
            )

    # Create and return the Pydantic model
    return create_model(model_name, **field_definitions)


def wrap_as_langchain_tool(tool: BaseTool) -> StructuredTool:
    """
    Wrap a custom BaseTool instance as a LangChain StructuredTool.

    This adapter allows our framework-agnostic tools to be used with LangGraph
    and other LangChain-based orchestrators.

    Args:
        tool: The BaseTool instance to wrap

    Returns:
        A LangChain StructuredTool that delegates to our tool
    """
    # Create Pydantic model from JSON Schema
    input_model = _json_schema_to_pydantic(
        tool.input_schema, model_name=f"{tool.name.capitalize().replace('_', '')}Input"
    )

    # Create async wrapper function
    async def tool_coroutine(**kwargs: Any) -> str:
        """Execute the tool and return result as string."""
        result = await tool.execute(dict(kwargs))

        # Return output or error
        if result.error:
            return f"Error: {result.error}"

        return str(result.output)

    # Create and return StructuredTool
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=input_model,
        coroutine=tool_coroutine,
    )


def wrap_tools_for_environment(registry: ToolRegistry, env: str) -> list[StructuredTool]:
    """
    Get all tools for an environment and wrap them as LangChain StructuredTools.

    This is a convenience function that combines registry lookup with LangChain
    wrapping. Use this in agent orchestrator setup.

    Args:
        registry: The ToolRegistry containing registered tools
        env: Environment name (e.g., "banking", "retail")

    Returns:
        List of LangChain StructuredTools scoped to the environment

    Example:
        ```python
        tools = wrap_tools_for_environment(registry, "banking")
        agent = create_react_agent(llm, tools)
        ```
    """
    base_tools = registry.list_for_environment(env)
    return [wrap_as_langchain_tool(tool) for tool in base_tools]
