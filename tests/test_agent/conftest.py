# tests/test_agent/conftest.py
"""Agent 层测试 fixtures."""

import pytest

from agent_layer.config import AgentConfig
from agent_layer.mcp.registry import ToolRegistry
from agent_layer.mcp.types import ToolRole
from agent_layer.llm.mock_client import MockLLMClient


@pytest.fixture
def agent_config():
    return AgentConfig(
        max_tool_rounds=3,
        enable_verification=True,
    )


@pytest.fixture
def registry():
    """创建一个注册了 mock 工具的 ToolRegistry."""
    reg = ToolRegistry()

    @reg.register(
        "get_stock_price",
        "获取股票行情",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码"},
                "start": {"type": "string", "description": "开始日期"},
                "end": {"type": "string", "description": "结束日期"},
            },
            "required": ["symbol", "start"],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_stock_price(symbol: str, start: str, end: str = "2024-12-31") -> dict:
        return {
            "symbol": symbol,
            "summary": {
                "symbol": symbol,
                "latest_close": 1850.50,
                "period": f"{start} ~ {end}",
                "count": 250,
            },
            "data": [
                {"date": "2024-12-31", "close": 1850.50, "volume": 5000000},
            ],
        }

    @reg.register(
        "search_stocks",
        "搜索股票",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "返回数量"},
            },
            "required": ["keyword"],
        },
        role=ToolRole.DATA_QUERY,
    )
    def search_stocks(keyword: str, limit: int = 10) -> dict:
        return {
            "keyword": keyword,
            "count": 2,
            "results": [
                {"symbol": "600519", "name": "贵州茅台", "market": "A股"},
                {"symbol": "000858", "name": "五粮液", "market": "A股"},
            ],
        }

    @reg.register(
        "search_knowledge",
        "知识库检索",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "top_k": {"type": "integer", "description": "返回数量"},
            },
            "required": ["query"],
        },
        role=ToolRole.RETRIEVAL,
    )
    def search_knowledge(query: str, top_k: int = 5) -> dict:
        return {
            "query": query,
            "count": 2,
            "results": [
                {
                    "score": 0.92,
                    "content": "贵州茅台是中国白酒行业龙头，ROE长期维持在30%以上。",
                    "metadata": {"doc_type": "financial_knowledge", "authority": 0.95},
                },
                {
                    "score": 0.85,
                    "content": "市盈率（PE）是估值常用指标，贵州茅台PE约为28倍。",
                    "metadata": {"doc_type": "financial_knowledge", "authority": 0.9},
                },
            ],
        }

    @reg.register(
        "get_macro_indicator",
        "获取宏观指标",
        parameters={
            "type": "object",
            "properties": {
                "indicator_name": {"type": "string", "description": "指标名称"},
            },
            "required": ["indicator_name"],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_macro_indicator(indicator_name: str) -> dict:
        return {
            "indicator": indicator_name,
            "latest": {"date": "2024-12-01", "value": 2.5},
            "data": [
                {"date": "2024-10-01", "value": 2.3},
                {"date": "2024-11-01", "value": 2.4},
                {"date": "2024-12-01", "value": 2.5},
            ],
        }

    return reg


@pytest.fixture
def mock_llm():
    return MockLLMClient()
