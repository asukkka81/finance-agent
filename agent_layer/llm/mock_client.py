# agent_layer/llm/mock_client.py
"""Mock LLM 客户端 — 用于测试 & 离线开发.

根据查询内容模拟 LLM 的工具调用决策，
不依赖真实的 LLM API。
"""

import json
import logging
import re
from typing import Optional

from agent_layer.llm.base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class MockLLMClient(BaseLLMClient):
    """Mock LLM 客户端.

    使用简单的规则匹配来模拟 LLM 的工具调用行为:
        - 查询含 "价格"/"行情"/"走势" → 调用 get_stock_price
        - 查询含 "什么是"/"如何"/"解释" → 调用 search_knowledge
        - 查询含 "搜索"/"找" → 调用 search_stocks
        - 复杂查询 → 多工具串行调用

    用于 Agent 层的单元测试，不需要真实 LLM API。
    """

    def __init__(self, **kwargs):
        self._call_count = 0  # 跟踪调用轮次

    def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """基本对话 — 直接返回最终答案."""
        last_msg = self._get_last_user_message(messages)
        return LLMResponse(
            content=f"Mock response to: {last_msg[:100]}...",
            finish_reason="stop",
        )

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """模拟带工具调用的决策.

        规则: 根据最后一条用户消息的内容决定调用哪些工具。
        """
        last_msg = self._get_last_user_message(messages)
        tool_names = {t["name"] for t in tools}
        tool_calls = self._decide_tools(last_msg, tool_names)

        self._call_count += 1

        if tool_calls:
            return LLMResponse(
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_use",
            )
        else:
            return LLMResponse(
                content=self._generate_answer(last_msg),
                finish_reason="stop",
            )

    def chat_with_tool_results(
        self,
        messages: list[dict],
        tool_results: list[dict],
        tools: list[dict],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """处理工具结果后: 决定是继续调用工具还是给出最终答案."""
        results_text = "\n".join(
            json.dumps(r, ensure_ascii=False, default=str)[:200]
            for r in tool_results
        )

        # 检查是否所有必要的数据都已获取
        has_price_data = any("get_stock_price" in json.dumps(r) for r in tool_results)
        has_knowledge = any("search_knowledge" in json.dumps(r) for r in tool_results)

        last_msg = self._get_last_user_message(messages)
        tool_names = {t["name"] for t in tools}

        # 策略:
        # 1. 如果有行情数据但没有知识背景 → 调 search_knowledge
        # 2. 如果有知识但没有行情 → 调 get_stock_price
        # 3. 两者都有 → 生成最终答案

        if has_price_data and not has_knowledge and "search_knowledge" in tool_names:
            self._call_count += 1
            return LLMResponse(
                content="",
                tool_calls=[self._make_call("search_knowledge", {
                    "query": last_msg,
                    "top_k": 5,
                })],
                finish_reason="tool_use",
            )

        if has_knowledge and not has_price_data and self._needs_price(last_msg):
            symbol = self._extract_symbol(last_msg)
            if symbol and "get_stock_price" in tool_names:
                self._call_count += 1
                return LLMResponse(
                    content="",
                    tool_calls=[self._make_call("get_stock_price", {
                        "symbol": symbol,
                        "start": "2024-01-01",
                        "end": "2024-12-31",
                    })],
                    finish_reason="tool_use",
                )

        # 最终答案
        return LLMResponse(
            content=self._generate_final_answer(last_msg, tool_results),
            finish_reason="stop",
        )

    # ================================================================
    # 规则引擎
    # ================================================================

    def _decide_tools(self, query: str, available: set[str]) -> list[dict]:
        """根据查询内容决定调用哪些工具."""
        calls = []

        # 股票行情相关
        if self._needs_price(query):
            symbol = self._extract_symbol(query)
            if symbol and "get_stock_price" in available:
                calls.append(self._make_call("get_stock_price", {
                    "symbol": symbol,
                    "start": "2024-01-01",
                    "end": "2024-12-31",
                }))

        # 知识检索相关
        if self._needs_knowledge(query) and "search_knowledge" in available:
            calls.append(self._make_call("search_knowledge", {
                "query": query,
                "top_k": 5,
            }))

        # 股票搜索
        if self._needs_stock_search(query) and "search_stocks" in available:
            calls.append(self._make_call("search_stocks", {
                "keyword": query,
                "limit": 10,
            }))

        # 宏观数据
        if self._needs_macro(query) and "get_macro_indicator" in available:
            indicator = self._extract_macro_indicator(query)
            calls.append(self._make_call("get_macro_indicator", {
                "indicator_name": indicator,
            }))

        return calls

    def _needs_price(self, query: str) -> bool:
        keywords = ["价格", "行情", "走势", "涨", "跌", "股价", "收盘", "K线",
                     "price", "stock price", "trend", "OHLCV"]
        return any(kw in query.lower() for kw in keywords)

    def _needs_knowledge(self, query: str) -> bool:
        keywords = ["什么是", "如何", "解释", "说明", "定义", "原理",
                     "PE", "ROE", "市盈率", "市净率", "估值",
                     "what is", "how to", "explain", "definition"]
        return any(kw in query.lower() for kw in keywords)

    def _needs_stock_search(self, query: str) -> bool:
        keywords = ["搜索", "查找", "找", "有哪些", "哪些股票", "列表",
                     "search", "find", "list", "which stocks"]
        return any(kw in query.lower() for kw in keywords)

    def _needs_macro(self, query: str) -> bool:
        keywords = ["CPI", "PMI", "GDP", "M2", "通胀", "利率", "宏观经济",
                     "inflation", "interest rate", "macro"]
        return any(kw in query.lower() for kw in keywords)

    def _extract_symbol(self, query: str) -> Optional[str]:
        """从查询中提取股票代码."""
        # 匹配 A股代码 (6位数字)
        m = re.search(r'\b(\d{6})\b', query)
        if m:
            return m.group(1)

        # 匹配美股代码 (大写字母 1-5 位)
        m = re.search(r'\b([A-Z]{1,5})\b', query)
        if m:
            return m.group(1)

        # 中文股票名 → 代码映射
        name_map = {
            "茅台": "600519", "贵州茅台": "600519",
            "五粮液": "000858",
            "苹果": "AAPL", "Apple": "AAPL",
            "微软": "MSFT", "Microsoft": "MSFT",
            "特斯拉": "TSLA", "Tesla": "TSLA",
            "英伟达": "NVDA", "Nvidia": "NVDA",
            "谷歌": "GOOGL", "Google": "GOOGL",
            "亚马逊": "AMZN", "Amazon": "AMZN",
            "宁德时代": "300750",
            "比亚迪": "002594",
            "招商银行": "600036",
        }
        for name, code in name_map.items():
            if name.lower() in query.lower():
                return code

        return None

    def _extract_macro_indicator(self, query: str) -> str:
        for kw in ["CPI", "cpi", "通胀", "消费者价格"]:
            if kw in query:
                return "cpi"
        for kw in ["PMI", "pmi", "采购经理"]:
            if kw in query:
                return "pmi"
        for kw in ["GDP", "gdp", "生产总值"]:
            if kw in query:
                return "gdp"
        for kw in ["M2", "m2", "货币供应"]:
            if kw in query:
                return "m2"
        return "cpi"  # 默认

    def _make_call(self, name: str, args: dict) -> dict:
        return {"id": f"mock_{name}_{self._call_count}", "name": name, "input": args}

    def _get_last_user_message(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    # Anthropic content block 格式
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            return block.get("text", "")
                    return str(content)
        return ""

    def _generate_answer(self, query: str) -> str:
        return f"根据您的查询「{query[:50]}...」，这是一个 Mock 响应。"

    def _generate_final_answer(self, query: str, results: list[dict]) -> str:
        parts = [f"综合回答您的查询「{query[:80]}...」："]
        for r in results:
            data = r.get("data", "")
            if isinstance(data, str) and len(data) < 200:
                parts.append(f"- {data}")
        return "\n".join(parts)
