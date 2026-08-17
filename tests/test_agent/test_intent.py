# tests/test_agent/test_intent.py
"""意图理解模块单元测试."""

import pytest

from agent_layer.core.intent import IntentParser, IntentType


class TestIntentParser:
    """测试意图解析器."""

    @pytest.fixture
    def parser(self):
        return IntentParser()

    def test_stock_price_intent(self, parser):
        """行情查询意图."""
        intent = parser.parse("贵州茅台最新股价是多少")
        assert intent.intent_type == IntentType.STOCK_PRICE
        assert "600519" in intent.symbols

    def test_knowledge_qa_intent(self, parser):
        """知识问答意图."""
        intent = parser.parse("什么是市盈率PE")
        assert intent.intent_type == IntentType.KNOWLEDGE_QA
        assert "市盈率" in intent.indicator_names

    def test_macro_intent(self, parser):
        """宏观分析意图."""
        intent = parser.parse("最近CPI走势怎么样")
        assert intent.intent_type == IntentType.MACRO_ANALYSIS
        assert "cpi" in intent.indicator_names

    def test_stock_search_intent(self, parser):
        """股票搜索意图."""
        intent = parser.parse("有哪些新能源股票")
        assert intent.intent_type == IntentType.STOCK_SEARCH

    def test_symbol_extraction_us(self, parser):
        """美股代码提取."""
        intent = parser.parse("AAPL股价走势")
        assert "AAPL" in intent.symbols

    def test_symbol_extraction_cn_name(self, parser):
        """中文名 → 代码映射."""
        # 五粮液
        intent = parser.parse("五粮液今天涨了吗")
        assert "000858" in intent.symbols

    def test_entity_indicator(self, parser):
        """指标提取."""
        intent = parser.parse("ROE是什么意思")
        assert "ROE" in intent.indicator_names

    def test_entity_market(self, parser):
        """市场识别."""
        intent = parser.parse("A股有哪些科技股")
        assert "CN" in intent.entities.get("market", [])

    def test_general_intent(self, parser):
        """通用意图."""
        intent = parser.parse("你好")
        assert intent.intent_type == IntentType.GENERAL
        assert intent.confidence < 0.5
