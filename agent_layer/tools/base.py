# agent_layer/tools/base.py
"""工具基类 & 装饰器."""

import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class BaseTool:
    """工具基类.

    提供跨工具的公共功能:
        - 参数校验
        - 错误处理
        - 结果截断 (防止 LLM 上下文溢出)
    """

    @staticmethod
    def validate_params(kwargs: dict, required: list[str]) -> Optional[str]:
        """校验必需参数. 返回错误信息或 None."""
        missing = [p for p in required if p not in kwargs or kwargs[p] is None]
        if missing:
            return f"缺少必需参数: {', '.join(missing)}"
        return None

    @staticmethod
    def truncate_result(data: Any, max_chars: int = 8000) -> Any:
        """截断过长的字符串结果 (防止 LLM 上下文溢出)."""
        if isinstance(data, str) and len(data) > max_chars:
            return data[:max_chars] + f"\n... (截断，原文共 {len(data)} 字符)"
        if isinstance(data, list) and len(data) > 50:
            return data[:50] + [f"... (截断，共 {len(data)} 条)"]
        return data

    @staticmethod
    def safe_call(fn: Callable, **kwargs) -> dict:
        """安全调用 — 统一异常处理."""
        try:
            result = fn(**kwargs)
            return {"success": True, "data": BaseTool.truncate_result(result)}
        except Exception as e:
            logger.error("Tool call failed: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}


def tool(name: str, description: str, parameters: dict):
    """工具装饰器 — 自动包装为 BaseTool.safe_call 风格.

    Usage::

        @tool("get_price", "获取价格", {...})
        def get_price(symbol: str, start: str, end: str) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(**kwargs) -> dict:
            return BaseTool.safe_call(func, **kwargs)

        wrapper._tool_name = name
        wrapper._tool_description = description
        wrapper._tool_parameters = parameters
        return wrapper

    return decorator
