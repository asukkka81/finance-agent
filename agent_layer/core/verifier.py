# agent_layer/core/verifier.py
"""多轮校验模块 — 对 Agent 输出进行四维度验证.

校验维度:
    1. 信息完整性 (completeness): 是否回答了用户的所有问题?
    2. 数据准确性 (accuracy): 引用的数据是否正确?
    3. 逻辑一致性 (logic): 推理链条是否自洽?
    4. 合规安全性 (compliance): 结论是否符合金融合规要求?

校验策略:
    - 每轮工具调用后检查信息完整性 → 不足则触发补充查询
    - 最终答案生成后综合四维度打分 → 不合格则触发修正
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agent_layer.prompts.templates import VERIFICATION_JUDGE_PROMPT

logger = logging.getLogger(__name__)


def format_tool_results_text(tool_results: list, max_chars: int = 3000) -> str:
    """将工具结果列表格式化为 LLM 可读文本 (供 judge / 重答 prompt 使用)."""
    parts = []
    for r in tool_results or []:
        if not isinstance(r, dict):
            continue
        name = r.get("tool_name", "unknown")
        data = r.get("data", {})
        try:
            text = json.dumps(data, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(data)
        parts.append(f"[{name}] {text[:600]}")
    return "\n".join(parts)[:max_chars]


# ================================================================
# 完整性校验: 实体提取 & 需求清单 (规则模式)
# ================================================================

# 常见指标/缩写 — 避免被误提取为美股股票代码
_STOCK_CODE_STOPWORDS = frozenset({
    "PE", "PB", "ROE", "ROA", "EPS", "CPI", "PMI", "GDP", "M2", "LPR",
    "AI", "ETF", "IPO", "MACD", "RSI", "OHLCV",
})

# 常见股票中文名 → 代码 (实体提取用)
_STOCK_NAME_MAP = {
    "贵州茅台": "600519", "茅台": "600519", "五粮液": "000858",
    "宁德时代": "300750", "比亚迪": "002594", "招商银行": "600036",
    "中国平安": "601318", "平安银行": "000001", "美的集团": "000333",
    "恒瑞医药": "600276", "紫金矿业": "601899",
    "苹果": "AAPL", "微软": "MSFT", "特斯拉": "TSLA", "英伟达": "NVDA",
    "谷歌": "GOOGL", "亚马逊": "AMZN",
}

# 需求清单: 查询关键词 → (需求名, 回答/数据中的覆盖关键词)
_REQUIREMENT_PATTERNS: dict[str, tuple[list[str], list[str]]] = {
    "估值水平": (
        ["估值", "市盈率", "PE", "市净率", "PB", "分位"],
        ["估值", "市盈率", "PE", "市净率", "PB", "分位", "倍"],
    ),
    "行业地位": (
        ["行业", "地位", "份额", "排名", "竞争", "龙头"],
        ["行业", "地位", "份额", "排名", "竞争", "龙头"],
    ),
    "对比分析": (
        ["对比", "比较", "相比", "区别", "哪个", "谁更", "vs"],
        ["对比", "相比", "高于", "低于", "优于", "区别", "vs"],
    ),
    "投资价值": (
        ["投资", "值得", "建仓", "买入", "价值"],
        ["投资", "价值", "回报", "收益", "值得", "买入", "持有"],
    ),
    "行情数据": (
        ["价格", "股价", "行情", "走势", "涨", "跌", "收盘", "现价", "多少"],
        ["价格", "收盘", "涨", "跌", "元", "点", "%"],
    ),
    "风险分析": (
        ["风险", "回撤", "波动"],
        ["风险", "回撤", "波动"],
    ),
    "基本面": (
        ["基本面", "财务", "营收", "利润", "ROE", "ROA"],
        ["营收", "利润", "ROE", "ROA", "基本面", "毛利率"],
    ),
    "宏观数据": (
        ["宏观", "CPI", "PMI", "GDP", "M2", "通胀", "利率"],
        ["CPI", "PMI", "GDP", "M2", "通胀", "利率"],
    ),
}


def extract_query_entities(query: str) -> list[dict]:
    """从查询中提取关键实体 (股票代码/中文名).

    Returns:
        [{"name": 实体显示名, "code": 代码}, ...] — 同名代码去重.
    """
    entities: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, code: str) -> None:
        if code not in seen:
            seen.add(code)
            entities.append({"name": name, "code": code})

    # A股 6 位数字代码
    for code in re.findall(r"\b(\d{6})\b", query):
        _add(code, code)

    # 中文股票名 (先匹配长名, 保证显示名更精确)
    for name, code in sorted(_STOCK_NAME_MAP.items(), key=lambda kv: -len(kv[0])):
        if name in query:
            _add(name, code)

    # 美股代码 (2-5 位大写, 排除指标缩写)
    for ticker in re.findall(r"\b([A-Z]{2,5})\b", query):
        if ticker not in _STOCK_CODE_STOPWORDS:
            _add(ticker, ticker)

    return entities


def _entity_covered(entity: dict, text: str) -> bool:
    """实体是否在答案/工具结果文本中被提及 (代码或任一中文名)."""
    if entity["code"] in text:
        return True
    return any(
        name in text
        for name, code in _STOCK_NAME_MAP.items()
        if code == entity["code"]
    )


class Verdict(str, Enum):
    PASS = "pass"         # 通过
    NEEDS_MORE = "needs_more"  # 信息不足，需要补充查询
    WEAK = "weak"         # 勉强通过但建议改进
    FAIL = "fail"         # 不通过，需要修正


@dataclass
class DimensionResult:
    """单维度校验结果."""

    dimension: str          # completeness / accuracy / logic / compliance
    verdict: Verdict
    score: float = 0.0     # 0~1
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """多维度校验综合结果.

    Attributes:
        passed: 是否全部通过.
        dimensions: 各维度结果.
        overall_score: 综合分数 0~1.
        needs_more_info: 是否需要补充查询.
        suggested_tool_calls: 建议的补充工具调用.
    """

    passed: bool = True
    dimensions: list[DimensionResult] = field(default_factory=list)
    overall_score: float = 0.0
    needs_more_info: bool = False
    suggested_tool_calls: list[dict] = field(default_factory=list)


class Verifier:
    """多轮校验器.

    completeness / compliance 使用规则校验;
    accuracy / logic 在启用 LLM 时由 LLM judge 评审 (核对数据引用与推理链条),
    judge 不可用时回退到规则检查。
    在校验失败时给出补充查询建议。
    """

    def __init__(self, use_llm: bool = False, llm_client=None):
        self.use_llm = use_llm
        self.llm_client = llm_client

    # ================================================================
    # 中间校验 (每轮工具调用后)
    # ================================================================

    def verify_after_tool_call(
        self,
        query: str,
        tool_results: list,
        intent_type: str,
    ) -> VerificationResult:
        """工具调用后的中间校验 — 检查信息完整性.

        判断: 当前收集的数据是否足以回答用户问题?

        Returns:
            VerificationResult — 如果 needs_more_info=True，
            suggested_tool_calls 中建议下一步调用的工具。
        """
        result = VerificationResult()

        # 1. 完整性检查: 是否覆盖了 intent 所需的所有数据?
        completeness = self._check_completeness(query, tool_results, intent_type)
        result.dimensions.append(completeness)

        if completeness.verdict == Verdict.NEEDS_MORE:
            result.needs_more_info = True
            result.suggested_tool_calls = completeness.suggestions
            result.passed = False

        # 2. 数据粗略检查: 是否有明显的错误?
        if tool_results:
            accuracy = self._quick_accuracy_check(tool_results)
            result.dimensions.append(accuracy)
            if accuracy.verdict == Verdict.FAIL:
                result.passed = False

        result.overall_score = sum(d.score for d in result.dimensions) / max(len(result.dimensions), 1)

        return result

    # ================================================================
    # 最终校验 (答案生成后)
    # ================================================================

    def verify_final_answer(
        self,
        query: str,
        answer: str,
        tool_results: list,
        intent_type: str,
    ) -> VerificationResult:
        """最终答案的四维度综合校验.

        accuracy / logic 维度在启用 LLM 时由 LLM judge 评审
        (核对答案引用的数据与工具结果、检查推理链条),
        调用失败时回退到规则检查。

        Returns:
            VerificationResult — 判断答案是否合规可用。
        """
        dimensions = []

        # LLM judge: 一次调用同时评审完整性 + 准确性 + 逻辑性
        judge = self._llm_judge(query, answer, tool_results)

        # 完整性
        dimensions.append(self._check_completeness(
            query, tool_results, intent_type, answer=answer, judge=judge,
        ))

        # 准确性
        dimensions.append(self._check_accuracy(answer, tool_results, judge))

        # 逻辑一致性
        dimensions.append(self._check_logic(answer, judge))

        # 合规安全性
        dimensions.append(self._check_compliance(answer, intent_type))

        scores = [d.score for d in dimensions]
        overall = sum(scores) / len(scores) if scores else 0.0
        # 如果所有维度分数都不低且没有 FAIL，应该通过
        if any(d.verdict == Verdict.FAIL for d in dimensions):
            all_pass = False
        else:
            all_pass = True  # NEEDS_MORE is not failure

        return VerificationResult(
            passed=all_pass,
            dimensions=dimensions,
            overall_score=overall,
        )

    # ================================================================
    # 各维度检查
    # ================================================================

    def _check_completeness(
        self,
        query: str,
        tool_results: list,
        intent_type: str,
        answer: str = "",
        judge: Optional[dict] = None,
    ) -> DimensionResult:
        """检查信息完整性.

        启用 LLM judge 且返回 completeness_score 时直接采用;
        否则规则校验, 三重检查:
            1. 实体覆盖: 查询中的股票代码/名称是否都在答案中被提及
            2. 需求清单: 查询关键词推导出的需求 (估值/行业/对比/投资...)
               是否都被回答
            3. 数据类型: 意图所需的数据类型是否已获取 (驱动补充查询)

        评分 = 0.5 * 实体覆盖率 + 0.5 * 需求覆盖率 - 0.2 * 缺失数据类型数
        """
        # LLM judge 模式
        if judge is not None and isinstance(judge.get("completeness_score"), (int, float)):
            score = self._clamp_score(judge.get("completeness_score"))
            missing = self._str_list(judge.get("missing_items"))
            return DimensionResult(
                dimension="completeness",
                verdict=self._verdict_from_score(score),
                score=score,
                issues=missing[:5] or self._str_list(judge.get("issues"))[:5],
            )

        issues: list[str] = []
        suggestions: list[dict] = []
        context_text = f"{answer}\n{format_tool_results_text(tool_results)}"

        # 1. 实体覆盖检查
        entities = extract_query_entities(query)
        missing_entities = [
            e["name"] for e in entities if not _entity_covered(e, context_text)
        ]
        entity_score = (
            (len(entities) - len(missing_entities)) / len(entities)
            if entities else 1.0
        )
        if missing_entities:
            issues.append(f"缺少实体分析: {', '.join(missing_entities)}")

        # 2. 需求清单检查
        requirements = self._extract_requirements(query)
        missing_requirements = [
            req_name
            for req_name, answer_kws in requirements.items()
            if not any(kw in context_text for kw in answer_kws)
        ]

        # 3. 数据类型覆盖检查 (原规则, 驱动补充查询)
        data_missing, data_flags = self._check_data_coverage(
            tool_results, intent_type, issues, suggestions,
        )

        # 数据驱动的需求 (行情/宏观) 只要有对应工具数据即视为已覆盖
        for req_name in list(missing_requirements):
            if req_name == "行情数据" and data_flags["has_price"]:
                missing_requirements.remove(req_name)
            elif req_name == "宏观数据" and data_flags["has_macro"]:
                missing_requirements.remove(req_name)

        req_score = (
            (len(requirements) - len(missing_requirements)) / len(requirements)
            if requirements else 1.0
        )
        if missing_requirements:
            issues.append(f"缺少需求分析: {', '.join(missing_requirements)}")

        score = 0.5 * entity_score + 0.5 * req_score - 0.2 * data_missing
        score = max(0.0, min(1.0, score))

        if data_missing and suggestions:
            verdict = Verdict.NEEDS_MORE
        else:
            verdict = self._verdict_from_score(score)

        return DimensionResult(
            dimension="completeness",
            verdict=verdict,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    @staticmethod
    def _extract_requirements(query: str) -> dict[str, list[str]]:
        """从查询关键词推导需求清单.

        Returns:
            {"需求名": [回答/数据中的覆盖关键词], ...}
        """
        query_upper = query.upper()
        requirements: dict[str, list[str]] = {}
        for req_name, (query_kws, answer_kws) in _REQUIREMENT_PATTERNS.items():
            if any(kw.upper() in query_upper for kw in query_kws):
                requirements[req_name] = answer_kws
        return requirements

    @staticmethod
    def _check_data_coverage(
        tool_results: list,
        intent_type: str,
        issues: list[str],
        suggestions: list[dict],
    ) -> tuple[int, dict]:
        """数据类型覆盖检查 (意图 → 必需数据).

        Returns:
            (缺失的数据类型数量, {"has_price"/"has_knowledge"/"has_macro": bool}).
        """
        tool_names = " ".join(str(r.get("tool_name", "")) for r in tool_results)
        flags = {
            "has_price": (
                "get_stock_price" in tool_names
                or "get_realtime_quote" in tool_names
            ),
            "has_knowledge": "search_knowledge" in tool_names,
            "has_macro": "get_macro_indicator" in tool_names,
        }
        missing = 0

        # 行情查询 → 需要价格数据
        if intent_type == "stock_price" and not flags["has_price"]:
            issues.append("缺少行情数据")
            suggestions.append({
                "tool": "get_stock_price",
                "reason": "需要获取股票行情数据",
            })
            missing += 1

        # 知识问答/投资建议 → 需要检索结果
        if intent_type in ("knowledge_qa", "investment_advice") and not flags["has_knowledge"]:
            issues.append("缺少专业知识背景")
            suggestions.append({
                "tool": "search_knowledge",
                "reason": "需要从知识库检索相关专业知识",
            })
            missing += 1

        # 宏观分析 → 需要宏观指标数据
        if intent_type == "macro_analysis" and not flags["has_macro"]:
            issues.append("缺少宏观数据")
            suggestions.append({
                "tool": "get_macro_indicator",
                "reason": "需要获取宏观指标数据",
            })
            missing += 1

        # 复合意图 → 行情与知识都需要
        if intent_type == "complex":
            if not flags["has_price"]:
                issues.append("缺少行情数据")
                suggestions.append({
                    "tool": "get_stock_price",
                    "reason": "获取行情数据",
                })
                missing += 1
            if not flags["has_knowledge"]:
                issues.append("缺少知识背景")
                suggestions.append({
                    "tool": "search_knowledge",
                    "reason": "检索专业知识",
                })
                missing += 1

        return missing, flags

    def _quick_accuracy_check(self, tool_results: list) -> DimensionResult:
        """快速数据准确性检查."""
        issues = []

        for r in tool_results:
            data = r.get("data", {})
            if isinstance(data, dict):
                # 检查是否有明显的错误标记
                if data.get("error"):
                    issues.append(f"工具返回错误: {data['error']}")

        if issues:
            return DimensionResult(
                dimension="accuracy",
                verdict=Verdict.FAIL,
                score=0.2,
                issues=issues,
            )

        return DimensionResult(
            dimension="accuracy",
            verdict=Verdict.PASS,
            score=0.9,
        )

    def _check_accuracy(
        self,
        answer: str,
        tool_results: list,
        judge: Optional[dict] = None,
    ) -> DimensionResult:
        """数据准确性检查 (最终).

        LLM judge 可用时: 核对答案引用的数字/日期与工具结果是否一致。
        否则回退规则检查 (空答案 / 工具错误 / 无数据支撑)。
        """
        if judge is not None:
            score = self._clamp_score(judge.get("accuracy_score"))
            return DimensionResult(
                dimension="accuracy",
                verdict=self._verdict_from_score(score),
                score=score,
                issues=self._str_list(judge.get("issues"))[:5],
                suggestions=self._str_list(judge.get("suggestions"))[:3],
            )

        # 规则回退
        if not answer or not answer.strip():
            return DimensionResult(
                dimension="accuracy", verdict=Verdict.FAIL, score=0.2,
                issues=["答案为空"],
            )
        if not tool_results:
            return DimensionResult(
                dimension="accuracy", verdict=Verdict.WEAK, score=0.5,
                issues=["无工具数据可供核对"],
            )
        for r in tool_results:
            data = r.get("data", {}) if isinstance(r, dict) else {}
            if isinstance(data, dict) and data.get("error"):
                return DimensionResult(
                    dimension="accuracy", verdict=Verdict.FAIL, score=0.2,
                    issues=[f"工具返回错误: {data['error']}"],
                )
        return DimensionResult(
            dimension="accuracy", verdict=Verdict.PASS, score=0.8,
        )

    def _check_logic(self, answer: str, judge: Optional[dict] = None) -> DimensionResult:
        """逻辑一致性检查 (最终).

        LLM judge 可用时: 评审推理链条是否自洽。
        否则回退规则检查 (仅能判断空答案, 深度逻辑校验依赖 LLM)。
        """
        if judge is not None:
            score = self._clamp_score(judge.get("logic_score"))
            return DimensionResult(
                dimension="logic",
                verdict=self._verdict_from_score(score),
                score=score,
                issues=self._str_list(judge.get("issues"))[:5],
                suggestions=self._str_list(judge.get("suggestions"))[:3],
            )

        # 规则回退
        if not answer or not answer.strip():
            return DimensionResult(
                dimension="logic", verdict=Verdict.FAIL, score=0.2,
                issues=["答案为空"],
            )
        return DimensionResult(
            dimension="logic", verdict=Verdict.WEAK, score=0.65,
            issues=["未启用 LLM，逻辑一致性未深度校验"],
        )

    # ================================================================
    # LLM Judge
    # ================================================================

    def _llm_judge(
        self,
        query: str,
        answer: str,
        tool_results: list,
    ) -> Optional[dict]:
        """调用 LLM 评审答案的准确性与逻辑性.

        Returns:
            {"accuracy_score": float, "logic_score": float,
             "issues": list[str], "suggestions": list[str]}
            或 None (未启用 / 调用失败 / 返回非法 JSON → 回退规则校验)。
        """
        if not (self.use_llm and self.llm_client):
            return None

        prompt = VERIFICATION_JUDGE_PROMPT.format(
            query=query,
            answer=answer,
            tool_results=format_tool_results_text(tool_results),
        )
        try:
            resp = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                system_prompt="你是严格的金融答案质量评审员，只输出 JSON。",
            )
        except Exception as e:
            logger.warning("LLM judge 调用失败, 回退规则校验: %s", e)
            return None

        judge = self._parse_judge_json(resp.content if resp else "")
        if judge is None:
            logger.warning(
                "LLM judge 返回非 JSON, 回退规则校验: %.80s",
                (resp.content if resp else "")[:80],
            )
        return judge

    @staticmethod
    def _parse_judge_json(content: str) -> Optional[dict]:
        """解析 judge 返回的 JSON (容忍前后多余文本)."""
        if not content:
            return None
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(data, dict):
            return None
        acc = data.get("accuracy_score")
        logic = data.get("logic_score")
        if not isinstance(acc, (int, float)) or not isinstance(logic, (int, float)):
            return None
        return {
            "accuracy_score": float(acc),
            "logic_score": float(logic),
            # 以下为可选字段 (老版 judge 可能不返回)
            "completeness_score": (
                float(data["completeness_score"])
                if isinstance(data.get("completeness_score"), (int, float)) else None
            ),
            "missing_items": data.get("missing_items") if isinstance(data.get("missing_items"), list) else [],
            "issues": data.get("issues") if isinstance(data.get("issues"), list) else [],
            "suggestions": data.get("suggestions") if isinstance(data.get("suggestions"), list) else [],
        }

    @staticmethod
    def _verdict_from_score(score: float) -> Verdict:
        """分数 → 判定: >=0.8 PASS, >=0.6 WEAK, 否则 FAIL."""
        if score >= 0.8:
            return Verdict.PASS
        if score >= 0.6:
            return Verdict.WEAK
        return Verdict.FAIL

    @staticmethod
    def _clamp_score(score) -> float:
        """将分数钳制到 0~1."""
        try:
            return max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _str_list(items) -> list[str]:
        """过滤出字符串元素."""
        if not isinstance(items, list):
            return []
        return [i for i in items if isinstance(i, str)]

    def _check_compliance(self, answer: str, intent_type: str) -> DimensionResult:
        """合规安全性检查."""
        issues = []

        # 检查是否包含投资建议的合规用语
        if intent_type in ("investment_advice", "complex"):
            # 必须包含风险提示
            risk_keywords = ["风险", "不构成", "仅供参考", "投资需谨慎", "过往业绩"]
            has_risk_warning = any(kw in answer for kw in risk_keywords)

            if not has_risk_warning and len(answer) > 50:
                issues.append("缺少投资风险提示")

        # 检查是否给出了具体买卖建议 (合规红线)
        buy_sell_patterns = ["建议买入", "建议卖出", "强烈推荐", "all in", "满仓"]
        for pattern in buy_sell_patterns:
            if pattern in answer:
                issues.append(f"包含不恰当的买卖建议: '{pattern}'")

        if issues:
            return DimensionResult(
                dimension="compliance",
                verdict=Verdict.FAIL if len(issues) > 1 else Verdict.WEAK,
                score=0.4,
                issues=issues,
            )

        return DimensionResult(
            dimension="compliance",
            verdict=Verdict.PASS,
            score=0.9,
        )
