# agent_layer/mcp/types.py
"""MCP 协议类型定义."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ToolRole(str, Enum):
    """工具角色 — 对应 Agent 五大工具体系."""

    DATA_QUERY = "data_query"           # 实时数据查询
    RETRIEVAL = "retrieval"             # 多模态检索
    WEB_FETCH = "web_fetch"             # 网页抓取
    TEXT2SQL = "text2sql"               # SQL 取数
    SANDBOX = "sandbox"                 # Python 沙箱


@dataclass
class ToolDefinition:
    """MCP 工具定义.

    对应 MCP 协议中的 Tool 概念:
        - name: 工具唯一标识
        - description: LLM 可读的功能描述
        - parameters: JSON Schema 格式的参数定义
    """

    name: str
    description: str
    parameters: dict  # JSON Schema
    role: ToolRole = ToolRole.DATA_QUERY
    handler: Optional[Callable] = None  # 实际执行函数

    def to_openai_tool(self) -> dict:
        """转为 OpenAI / Anthropic tool-use 格式."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters.get("properties", {}),
                "required": self.parameters.get("required", []),
            },
        }


@dataclass
class ToolCall:
    """一次工具调用请求.

    LLM 发出工具调用 → Agent 解析为此对象 → 执行 handler.
    """

    tool_name: str
    arguments: dict
    call_id: str = ""  # 调用的唯一 ID (用于关联响应)

    @classmethod
    def from_llm_response(cls, response: dict) -> "ToolCall":
        """从 LLM tool_use block 解析."""
        return cls(
            tool_name=response.get("name", ""),
            arguments=response.get("input", {}),
            call_id=response.get("id", ""),
        )


@dataclass
class ToolResult:
    """工具调用结果."""

    tool_name: str
    call_id: str = ""
    success: bool = True
    data: Any = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转为 LLM 可读的 tool_result 格式."""
        return {
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }

    def summary(self, max_len: int = 200) -> str:
        """生成结果摘要 (用于 LLM 上下文中)."""
        if not self.success:
            return f"[ERROR] {self.tool_name}: {self.error}"

        data_str = str(self.data)
        if len(data_str) > max_len:
            data_str = data_str[:max_len] + f"... (truncated, total {len(data_str)} chars)"

        return f"[OK] {self.tool_name}: {data_str}"
