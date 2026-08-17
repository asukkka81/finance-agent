# agent_layer/tools/__init__.py
"""Agent 工具集 — 封装数据层 & 检索层为 MCP 标准工具."""

from agent_layer.tools.base import BaseTool, tool
from agent_layer.tools.data_tools import register_data_tools
from agent_layer.tools.retrieval_tools import register_retrieval_tools
from agent_layer.tools.web_tools import register_web_tools

__all__ = [
    "BaseTool",
    "tool",
    "register_data_tools",
    "register_retrieval_tools",
    "register_web_tools",
]
