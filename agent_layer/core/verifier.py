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

        # LLM judge: 一次调用同时评审准确性 + 逻辑性
        judge = self._llm_judge(query, answer, tool_results)

        # 完整性
        dimensions.append(self._check_completeness(query, tool_results, intent_type))

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
