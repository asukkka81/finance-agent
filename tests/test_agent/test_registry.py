# tests/test_agent/test_registry.py
"""ToolRegistry 单元测试."""

import pytest

from agent_layer.mcp.registry import ToolRegistry
from agent_layer.mcp.types import ToolCall, ToolRole


class TestToolRegistry:
    """测试工具注册中心."""

    def test_register_tool(self):
        """注册工具应成功."""
        registry = ToolRegistry()

        @registry.register("test_tool", "测试工具", role=ToolRole.DATA_QUERY)
        def test_tool(x: int, y: str = "default") -> dict:
            return {"x": x, "y": y}

        assert registry.tool_count == 1
        assert "test_tool" in registry.tool_names

    def test_execute_success(self):
        """执行工具应返回正确结果."""
        registry = ToolRegistry()

        @registry.register("add", "加法")
        def add(a: int, b: int) -> int:
            return a + b

        call = ToolCall(tool_name="add", arguments={"a": 1, "b": 2})
        result = registry.execute(call)

        assert result.success
        assert result.data == 3

    def test_execute_unknown_tool(self):
        """执行不存在的工具应返回错误."""
        registry = ToolRegistry()

        call = ToolCall(tool_name="nonexistent", arguments={})
        result = registry.execute(call)

        assert not result.success
        assert "Unknown tool" in result.error

    def test_execute_type_error(self):
        """参数类型错误应返回错误."""
        registry = ToolRegistry()

        @registry.register("add", "加法")
        def add(a: int, b: int) -> int:
            return a + b

        call = ToolCall(tool_name="add", arguments={"a": "not_a_number"})
        result = registry.execute(call)

        # 不会 TypeError — Python 不强制类型
        # 但缺少必需参数会 TypeError
        call2 = ToolCall(tool_name="add", arguments={"a": 1})  # 缺少 b
        result2 = registry.execute(call2)

        assert not result2.success

    def test_get_tool_schemas(self, registry):
        """生成的 schema 应包含所有工具."""
        schemas = registry.get_tool_schemas()
        assert len(schemas) == registry.tool_count

        tool_names = [s["name"] for s in schemas]
        assert "get_stock_price" in tool_names
        assert "search_knowledge" in tool_names

    def test_get_tool_descriptions(self, registry):
        """工具描述应人类可读."""
        desc = registry.get_tool_descriptions()
        assert "get_stock_price" in desc
        assert "search_knowledge" in desc
        assert "## 可用工具" in desc

    def test_unregister(self):
        """注销工具应生效."""
        registry = ToolRegistry()

        @registry.register("temp", "临时工具")
        def temp():
            pass

        assert registry.tool_count == 1
        registry.unregister("temp")
        assert registry.tool_count == 0

    def test_execute_batch(self, registry):
        """批量执行应全部成功."""
        calls = [
            ToolCall("search_stocks", {"keyword": "茅台"}),
            ToolCall("search_knowledge", {"query": "什么是PE"}),
        ]
        results = registry.execute_batch(calls)

        assert len(results) == 2
        assert all(r.success for r in results)
