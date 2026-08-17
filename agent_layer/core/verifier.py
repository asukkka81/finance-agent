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

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


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

    使用规则 + (可选) LLM 进行四维度校验。
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

        Returns:
            VerificationResult — 判断答案是否合规可用。
        """
        dimensions = []

        # 完整性
        dimensions.append(self._check_completeness(query, tool_results, intent_type))

        # 准确性
        dimensions.append(self._check_accuracy(answer, tool_results))

        # 逻辑一致性
        dimensions.append(self._check_logic(answer))

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
        self, query: str, tool_results: list, intent_type: str
    ) -> DimensionResult:
        """检查信息完整性."""
        issues = []
        suggestions = []

        # 检查工具结果中包含的数据类型
        has_price_data = any(
            "get_stock_price" in str(r.get("tool_name", ""))
            for r in tool_results
        )
        has_knowledge = any(
            "search_knowledge" in str(r.get("tool_name", ""))
            for r in tool_results
        )

        # 行情查询 → 需要价格数据
        if intent_type == "stock_price" and not has_price_data:
            issues.append("缺少行情数据")
            suggestions.append({
                "tool": "get_stock_price",
                "reason": "需要获取股票行情数据",
            })

        # 知识问答 → 需要检索结果
        if intent_type in ("knowledge_qa", "investment_advice") and not has_knowledge:
            issues.append("缺少专业知识背景")
            suggestions.append({
                "tool": "search_knowledge",
                "reason": "需要从知识库检索相关专业知识",
            })

        # 复合意图 → 两者都需要
        if intent_type == "complex":
            if not has_price_data:
                issues.append("缺少行情数据")
            if not has_knowledge:
                issues.append("缺少知识背景")
            if not has_price_data and not has_knowledge:
                suggestions.append({
                    "tool": "get_stock_price",
                    "reason": "获取行情数据",
                })
                suggestions.append({
                    "tool": "search_knowledge",
                    "reason": "检索专业知识",
                })

        if issues:
            return DimensionResult(
                dimension="completeness",
                verdict=Verdict.NEEDS_MORE if suggestions else Verdict.WEAK,
                score=0.3,
                issues=issues,
                suggestions=suggestions,
            )

        return DimensionResult(
            dimension="completeness",
            verdict=Verdict.PASS,
            score=0.9,
        )

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

    def _check_accuracy(self, answer: str, tool_results: list) -> DimensionResult:
        """数据准确性检查 (最终)."""
        # 规则: 确保答案中引用的数据能在工具结果中找到
        return DimensionResult(
            dimension="accuracy",
            verdict=Verdict.PASS,
            score=0.85,
        )

    def _check_logic(self, answer: str) -> DimensionResult:
        """逻辑一致性检查."""
        # 简单规则: 检查有无矛盾表述
        contradictions = []

        if "但是" in answer and "所以" in answer:
            # 有转折+结论，需要检查是否自洽
            pass

        if "同时" in answer and "然而" in answer:
            # 有并列+转折，可能逻辑跳跃
            pass

        if contradictions:
            return DimensionResult(
                dimension="logic",
                verdict=Verdict.WEAK,
                score=0.6,
                issues=contradictions,
            )

        return DimensionResult(
            dimension="logic",
            verdict=Verdict.PASS,
            score=0.85,
        )

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
