# agent_layer/mcp/__init__.py
"""MCP 协议 — 工具注册、调用、结果标准化."""

from agent_layer.mcp.types import ToolDefinition, ToolCall, ToolResult, ToolRole
from agent_layer.mcp.registry import ToolRegistry

__all__ = [
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolRole",
    "ToolRegistry",
]
