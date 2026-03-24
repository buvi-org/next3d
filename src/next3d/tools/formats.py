"""Export tool schemas in OpenAI function-calling and MCP formats.

AI frameworks expect tool definitions in specific JSON formats.
This module converts our Pydantic tool schemas into those formats
so any AI agent can discover and call our 3D modeling tools.
"""

from __future__ import annotations

import json
from typing import Any

from next3d.tools.schema import TOOL_SCHEMAS


def to_openai_tools() -> list[dict[str, Any]]:
    """Export all tools in OpenAI function-calling format.

    Returns a list of tool definitions compatible with the
    OpenAI Chat Completions API `tools` parameter.
    """
    tools = []
    for name, schema_cls in TOOL_SCHEMAS.items():
        json_schema = schema_cls.model_json_schema()
        # Remove pydantic metadata that OpenAI doesn't need
        json_schema.pop("title", None)

        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": schema_cls.__doc__ or name,
                "parameters": json_schema,
            },
        }
        tools.append(tool)
    return tools


def to_mcp_tools() -> list[dict[str, Any]]:
    """Export all tools in MCP (Model Context Protocol) format.

    Returns a list of tool definitions compatible with the
    MCP tools/list response.
    """
    tools = []
    for name, schema_cls in TOOL_SCHEMAS.items():
        json_schema = schema_cls.model_json_schema()
        json_schema.pop("title", None)

        tool = {
            "name": name,
            "description": schema_cls.__doc__ or name,
            "inputSchema": json_schema,
        }
        tools.append(tool)
    return tools


def to_anthropic_tools() -> list[dict[str, Any]]:
    """Export all tools in Anthropic Claude API format.

    Returns a list of tool definitions compatible with the
    Anthropic Messages API `tools` parameter.
    """
    tools = []
    for name, schema_cls in TOOL_SCHEMAS.items():
        json_schema = schema_cls.model_json_schema()
        json_schema.pop("title", None)

        tool = {
            "name": name,
            "description": schema_cls.__doc__ or name,
            "input_schema": json_schema,
        }
        tools.append(tool)
    return tools


def to_json(fmt: str = "openai") -> str:
    """Export tool schemas as JSON string.

    Args:
        fmt: Format — "openai", "mcp", or "anthropic".
    """
    exporters = {
        "openai": to_openai_tools,
        "mcp": to_mcp_tools,
        "anthropic": to_anthropic_tools,
    }
    exporter = exporters.get(fmt)
    if not exporter:
        raise ValueError(f"Unknown format: {fmt}. Use: {', '.join(exporters)}")
    return json.dumps(exporter(), indent=2)
