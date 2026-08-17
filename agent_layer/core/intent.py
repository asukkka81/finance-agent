# agent_layer/core/intent.py
"""意图理解模块 — 将用户自然语言查询解析为结构化意图.

解析维度:
    1. 意图类型: 行情查询 / 知识问答 / 股票搜索 / 宏观分析 / 投资建议 / 组合分析
    2. 关键实体: 股票代码、指标名称、时间范围、行业、基金...
    3. 约束条件: 时间范围、市场限制、精度要求...
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """用户意图分类."""

    STOCK_PRICE = "stock_price"         # 行情查询: "茅台股价多少"
    STOCK_SEARCH = "stock_search"       # 股票搜索: "有哪些新能源股票"
    KNOWLEDGE_QA = "knowledge_qa"       # 知识问答: "什么是PE"
    MACRO_ANALYSIS = "macro_analysis"   # 宏观分析: "CPI走势"
    FUND_QUERY = "fund_query"           # 基金查询: "沪深300ETF净值"
    INVESTMENT_ADVICE = "investment_advice"  # 投资建议: "如何配置资产"
    PORTFOLIO = "portfolio"             # 组合分析: "我的持仓收益"
    COMPLEX = "complex"                 # 复合意图: 需要多工具协作
    GENERAL = "general"                 # 通用闲聊


@dataclass
class Intent:
    """解析后的用户意图.

    Attributes:
        intent_type: 意图类型.
        entities: 提取的关键实体 {type: value}.
        original_query: 用户原始输入.
        confidence: 意图识别置信度 0~1.
        constraints: 时间/数量等约束.
    """

    intent_type: IntentType = IntentType.GENERAL
    entities: dict[str, list[str]] = field(default_factory=dict)
    original_query: str = ""
    confidence: float = 0.0
    constraints: dict = field(default_factory=dict)

    @property
    def symbols(self) -> list[str]:
        """提取的所有股票代码."""
        return self.entities.get("symbol", [])

    @property
    def indicator_names(self) -> list[str]:
        """提取的指标名称."""
        return self.entities.get("indicator", [])

    @property
    def time_range(self) -> Optional[tuple[str, str]]:
        """时间范围 (start, end)."""
        return self.constraints.get("time_range")


class IntentParser:
    """意图解析器 — 基于规则 + 关键实体提取.

    在 Qwen 微调之前使用规则引擎，微调后替换为模型推理。
    """

    # 意图关键词模式
    INTENT_PATTERNS = {
        IntentType.MACRO_ANALYSIS: [  # MACRO FIRST — higher priority
            r'\b(CPI|PMI|GDP|M2|LPR|通胀|利率|宏观)\b',
            r'\b(inflation|interest rate|macro|economy)\b',
        ],
        IntentType.INVESTMENT_ADVICE: [  # ADVICE before KNOWLEDGE
            r'(建议|推荐|配置|策略|理财|投资组合|资产配置|仓位|定投)',
            r'(advice|recommend|portfolio|strategy|allocation|invest)',
        ],
        IntentType.STOCK_PRICE: [
            r'(股价|行情|价格|走势|涨|跌|收盘|开盘|K线|多少[钱点])',
            r'(price|trend|quote|how much is)',
        ],
        IntentType.STOCK_SEARCH: [
            r'(搜索|查找|找|有哪些|哪些股票|列表|筛选)',
            r'(search|find|list|which stocks|screen)',
        ],
        IntentType.KNOWLEDGE_QA: [
            r'(什么是|什么叫|如何|怎么|为什么|解释|说明|定义|原理)',
            r'(what is|how to|explain|definition|meaning)',
        ],
        IntentType.MACRO_ANALYSIS: [
            r'(CPI|PMI|GDP|M2|通胀|利率|LPR|宏观)',
            r'(inflation|interest rate|macro|economy)',
        ],
        IntentType.FUND_QUERY: [
            r'(基金|净值|ETF|LOF|QDII)',
            r'(fund|NAV|ETF)',
        ],
        IntentType.INVESTMENT_ADVICE: [
            r'(建议|推荐|配置|策略|理财|投资组合)',
            r'(advice|recommend|portfolio|strategy|allocation)',
        ],
        IntentType.PORTFOLIO: [
            r'(持仓|我的[仓位]|盈亏|收益|仓位|仓位分析)',
            r'(portfolio|position|holding|pnl|return)',
        ],
    }

    # 实体提取模式
    ENTITY_PATTERNS = {
        "symbol": [
            # A股 6 位代码
            r'(?<![0-9])(\d{6})(?![0-9])',
            # 美股代码 (1-5 位大写字母，可含连字符)
            r'(?<![A-Za-z])([A-Z]{1,5})(?:-[A-Z])?(?![A-Za-z])',
            # 中文名 → 代码
            (r'(?:贵州){0,1}茅台', '600519'),
            (r'五粮液', '000858'),
            (r'招商银行', '600036'),
            (r'中国平安', '601318'),
            (r'宁德时代', '300750'),
            (r'比亚迪', '002594'),
            (r'Apple|苹果', 'AAPL'),
            (r'Microsoft|微软', 'MSFT'),
            (r'Tesla|特斯拉', 'TSLA'),
        ],
        "indicator": [
            (r'CPI|cpi|消费者价格|通胀率', 'cpi'),
            (r'PMI|pmi|采购经理', 'pmi'),
            (r'GDP|gdp|生产总值', 'gdp'),
            (r'M2|m2|货币供应', 'm2'),
            (r'LPR|lpr|贷款市场报价', 'lpr'),
            (r'PE|pe|市盈率', '市盈率'),
            (r'ROE|roe|净资产收益率', 'ROE'),
            (r'ROA|roa|总资产收益率', 'ROA'),
        ],
        "market": [
            (r'A股|a股|中国|沪深|上证|深证', 'CN'),
            (r'美股|纳斯达克|纽交所|NYSE|NASDAQ', 'US'),
        ],
    }

    def parse(self, query: str) -> Intent:
        """解析用户查询为结构化意图.

        Args:
            query: 用户自然语言输入.

        Returns:
            Intent 对象.
        """
        intent_type, confidence = self._classify_intent(query)
        entities = self._extract_entities(query)
        constraints = self._extract_constraints(query)

        intent = Intent(
            intent_type=intent_type,
            entities=entities,
            original_query=query,
            confidence=confidence,
            constraints=constraints,
        )

        logger.debug(
            "Parsed intent: type=%s confidence=%.2f entities=%s",
            intent_type.value, confidence, entities,
        )
        return intent

    def _classify_intent(self, query: str) -> tuple[IntentType, float]:
        """分类意图类型."""
        scores: dict[IntentType, float] = {}

        for intent_type, patterns in self.INTENT_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1.0
            if score > 0:
                scores[intent_type] = score / len(patterns)

        if not scores:
            return IntentType.GENERAL, 0.3

        # 返回得分最高的意图
        best = max(scores, key=scores.get)
        return best, min(scores[best], 1.0)

    def _extract_entities(self, query: str) -> dict[str, list[str]]:
        """提取关键实体."""
        entities: dict[str, list[str]] = {}

        # Symbol 提取
        symbols = set()
        for pattern in self.ENTITY_PATTERNS.get("symbol", []):
            if isinstance(pattern, tuple):
                search_pat, code = pattern
                if re.search(search_pat, query, re.IGNORECASE):
                    symbols.add(code)
            else:
                for match in re.finditer(pattern, query, re.IGNORECASE):
                    symbols.add(match.group(1))

        if symbols:
            entities["symbol"] = list(symbols)

        # Indicator 提取
        indicators = set()
        for pattern in self.ENTITY_PATTERNS.get("indicator", []):
            if isinstance(pattern, tuple):
                search_pat, ind_name = pattern
                if re.search(search_pat, query, re.IGNORECASE):
                    indicators.add(ind_name)

        if indicators:
            entities["indicator"] = list(indicators)

        # Market 提取
        markets = set()
        for pattern in self.ENTITY_PATTERNS.get("market", []):
            if isinstance(pattern, tuple):
                search_pat, mkt = pattern
                if re.search(search_pat, query, re.IGNORECASE):
                    markets.add(mkt)

        if markets:
            entities["market"] = list(markets)

        return entities

    def _extract_constraints(self, query: str) -> dict:
        """提取约束条件 (时间、数量等)."""
        constraints = {}

        # 时间范围
        date_patterns = [
            (r'(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日号]?', 'specific'),
            (r'最近(\d+)[个]?[天日月]', 'relative_days'),
            (r'(\d{4})年', 'year'),
            (r'本[周月季度年]', 'current_period'),
        ]

        for pattern, ptype in date_patterns:
            match = re.search(pattern, query)
            if match and ptype == 'specific':
                y, m, d = match.groups()
                constraints["time_range"] = (
                    f"{y}-{m.zfill(2)}-{d.zfill(2)}",
                    "today",
                )
                constraints["time_type"] = "specific"
                break

        # 数量限制
        num_match = re.search(r'(前|top)?\s*(\d+)\s*[个条只]', query)
        if num_match:
            constraints["limit"] = int(num_match.group(2))

        return constraints
