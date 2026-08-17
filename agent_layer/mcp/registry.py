# agent_layer/mcp/registry.py
"""MCP 工具注册中心 — 工具发现、调用、Schema 生成."""

import logging
from typing import Callable, Optional

from agent_layer.mcp.types import ToolCall, ToolDefinition, ToolResult, ToolRole

logger = logging.getLogger(__name__)


class ToolRegistry:
    """MCP 工具注册中心.

    管理所有可用工具的生命周期:
        - 注册 / 注销
        - 生成 LLM tool-use schema
        - 执行工具调用并返回标准化结果

    Usage::

        registry = ToolRegistry()

        @registry.register("get_stock_price", "获取股票实时行情", role=ToolRole.DATA_QUERY)
        async def get_stock_price(symbol: str, start: str, end: str) -> dict:
            ...

        # 执行
        result = registry.execute(ToolCall(tool_name="get_stock_price", arguments={...}))
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    # ================================================================
    # 工具注册
    # ================================================================

    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[dict] = None,
        role: ToolRole = ToolRole.DATA_QUERY,
    ) -> Callable:
        """装饰器: 注册一个工具.

        Usage::

            @registry.register(
                "get_stock_price",
                "获取指定股票在日期范围内的日线行情数据",
                parameters={
                    "properties": {
                        "symbol": {"type": "string", "description": "股票代码"},
                        "start": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
                        "end": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
                    },
                    "required": ["symbol", "start", "end"],
                },
                role=ToolRole.DATA_QUERY,
            )
            def get_stock_price(symbol, start, end):
                ...
        """
        def decorator(handler: Callable):
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters or self._infer_parameters(handler),
                role=role,
                handler=handler,
            )
            logger.info("Registered tool: %s (role=%s)", name, role.value)
            return handler

        return decorator

    def register_tool(self, definition: ToolDefinition) -> None:
        """直接注册一个 ToolDefinition 对象."""
        self._tools[definition.name] = definition
        logger.info("Registered tool: %s", definition.name)

    def unregister(self, name: str) -> None:
        """注销工具."""
        self._tools.pop(name, None)

    # ================================================================
    # 工具调用
    # ================================================================

    def execute(self, call: ToolCall) -> ToolResult:
        """执行工具调用.

        Args:
            call: ToolCall 对象 (来自 LLM tool_use).

        Returns:
            ToolResult 包含执行结果或错误信息.
        """
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return ToolResult(
                tool_name=call.tool_name,
                call_id=call.call_id,
                success=False,
                error=f"Unknown tool: {call.tool_name}. Available: {self.tool_names}",
            )

        if tool.handler is None:
            return ToolResult(
                tool_name=call.tool_name,
                call_id=call.call_id,
                success=False,
                error=f"Tool '{call.tool_name}' has no handler",
            )

        try:
            data = tool.handler(**call.arguments)
            return ToolResult(
                tool_name=call.tool_name,
                call_id=call.call_id,
                success=True,
                data=data,
            )
        except TypeError as e:
            return ToolResult(
                tool_name=call.tool_name,
                call_id=call.call_id,
                success=False,
                error=f"Parameter error: {e}",
            )
        except Exception as e:
            logger.error(
                "Tool '%s' failed: %s", call.tool_name, e, exc_info=True
            )
            return ToolResult(
                tool_name=call.tool_name,
                call_id=call.call_id,
                success=False,
                error=str(e),
            )

    def execute_batch(self, calls: list[ToolCall]) -> list[ToolResult]:
        """批量执行工具调用 (顺序)."""
        return [self.execute(c) for c in calls]

    # ================================================================
    # LLM 接口
    # ================================================================

    def get_tool_schemas(self) -> list[dict]:
        """生成所有工具的 schema 列表 (供 LLM tool-use API 使用).

        Returns:
            Anthropic-compatible tool definitions.
        """
        return [
            tool.to_openai_tool()
            for tool in self._tools.values()
        ]

    def get_tool_descriptions(self) -> str:
        """生成工具列表的人类可读描述 (供 prompt engineering 使用)."""
        lines = ["## 可用工具\n"]
        for tool in self._tools.values():
            params_desc = tool.parameters.get("properties", {})
            param_strs = []
            for pname, pinfo in params_desc.items():
                required = pname in tool.parameters.get("required", [])
                marker = "*" if required else ""
                param_strs.append(f"  {marker}{pname} ({pinfo.get('type', 'any')}): {pinfo.get('description', '')}")

            lines.append(f"### {tool.name} [{tool.role.value}]")
            lines.append(f"{tool.description}")
            if param_strs:
                lines.append("参数:")
                lines.extend(param_strs)
            lines.append("")

        return "\n".join(lines)

    # ================================================================
    # 属性
    # ================================================================

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    # ================================================================
    # 内部
    # ================================================================

    @staticmethod
    def _infer_parameters(handler: Callable) -> dict:
        """从函数签名推断参数 schema (默认全为 string)."""
        import inspect

        sig = inspect.signature(handler)
        properties = {}
        required = []

        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            param_type = "string"
            if param.annotation is not inspect.Parameter.empty:
                annotation = param.annotation
                if annotation is int:
                    param_type = "integer"
                elif annotation is float:
                    param_type = "number"
                elif annotation is bool:
                    param_type = "boolean"

            properties[pname] = {
                "type": param_type,
                "description": f"{pname} 参数",
            }

            if param.default is inspect.Parameter.empty:
                required.append(pname)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
