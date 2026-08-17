# model_layer/evaluation/metrics.py
"""模型评估指标 — 金融 Agent 专用评测维度.

指标:
    1. 工具调用准确率 (Tool Call Accuracy)
    2. 答案合规率 (Compliance Rate)
    3. 幻觉率 (Hallucination Rate)
    4. 代码执行成功率 (Code Execution Success)
    5. BLEU / ROUGE (标准文本指标)
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def compute_all_metrics(
    predictions: list[dict],
    references: list[dict],
) -> dict:
    """计算全部评估指标.

    Args:
        predictions: [{"response": str, "tool_calls": [...], "code": str}, ...]
        references: [{"expected_tools": [...], "expected_answer": str, ...}, ...]

    Returns:
        {metric_name: score} 字典.
    """
    return {
        "tool_call_accuracy": compute_tool_accuracy(predictions, references),
        "compliance_rate": compute_compliance_rate(
            [p.get("response", "") for p in predictions]
        ),
        "hallucination_score": compute_hallucination_score(
            [p.get("response", "") for p in predictions],
            [r.get("expected_facts", []) for r in references],
        ),
        "code_success_rate": compute_code_success_rate(
            [p.get("code_result", {}) for p in predictions]
        ),
        "format_compliance": compute_format_compliance(
            [p.get("response", "") for p in predictions]
        ),
        "tool_efficiency": compute_tool_efficiency(
            [p.get("tool_calls", []) for p in predictions],
            [r.get("expected_tools", []) for r in references],
        ),
    }


# ================================================================
# 各项指标
# ================================================================


def compute_tool_accuracy(
    predictions: list[dict],
    references: list[dict],
) -> float:
    """工具调用准确率 — 是否调用了正确的工具.

    Returns:
        0~1 分数.
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, ref in zip(predictions, references):
        pred_tools = {tc.get("name", "") for tc in pred.get("tool_calls", [])}
        expected_tools = set(ref.get("expected_tools", []))

        if not expected_tools:
            correct += 1 if not pred_tools else 0
            continue

        # 精确匹配 + 部分匹配
        intersection = pred_tools & expected_tools
        if len(intersection) == len(expected_tools):
            correct += 1
        elif len(intersection) > 0:
            correct += 0.5

    return correct / len(predictions)


def compute_compliance_rate(responses: list[str]) -> float:
    """答案合规率 — 是否包含必要的风险提示和合规用语.

    Returns:
        0~1 分数 (越高越合规).
    """
    if not responses:
        return 0.0

    compliant = 0
    required_phrases = ["风险", "不构成", "仅供参考", "投资需谨慎"]

    for resp in responses:
        if len(resp) < 50:
            compliant += 1  # 短回答无需合规用语
            continue

        found = sum(1 for phrase in required_phrases if phrase in resp)
        if found >= 2:
            compliant += 1
        elif found >= 1:
            compliant += 0.5

    return compliant / len(responses)


def compute_hallucination_score(
    responses: list[str],
    expected_facts: list[list[str]],
) -> float:
    """幻觉评分 — 检查回答中是否包含不存在的事实.

    简化版: 检查 generated text 中关键指标的数值是否合理。

    Returns:
        0~1 分数 (越高幻觉越少).
    """
    if not responses:
        return 0.0

    scores = []
    for resp, facts in zip(responses, expected_facts):
        if not facts:
            scores.append(0.8)  # 无参考事实，默认中等分
            continue

        # 检查每个预期事实是否在回答中
        found = sum(1 for fact in facts if fact.lower() in resp.lower())
        scores.append(found / len(facts))

    return sum(scores) / len(scores)


def compute_code_success_rate(code_results: list[dict]) -> float:
    """代码执行成功率.

    Returns:
        0~1 分数.
    """
    if not code_results:
        return 1.0  # 无代码，默认成功

    success = sum(1 for r in code_results if r.get("success", False))
    return success / len(code_results)


def compute_format_compliance(responses: list[str]) -> float:
    """格式合规 — 是否按约定的结构组织回答.

    Returns:
        0~1 分数.
    """
    if not responses:
        return 0.0

    scores = []
    for resp in responses:
        score = 1.0

        # 检查是否有标题/分段
        if not any(marker in resp for marker in ["**", "##", "###", "1.", ">"]):
            score -= 0.2

        # 检查是否有数据引用
        has_numbers = bool(re.search(r'\d+\.?\d*[%倍元亿万亿]', resp))
        if not has_numbers and len(resp) > 80:
            score -= 0.1

        scores.append(max(0.0, score))

    return sum(scores) / len(scores)


def compute_tool_efficiency(
    pred_tool_calls: list[list[dict]],
    expected_tools: list[list[str]],
) -> float:
    """工具调用效率 — 避免冗余调用.

    Returns:
        0~1 分数 (越高效率越高).
    """
    if not pred_tool_calls:
        return 1.0

    scores = []
    for calls, expected in zip(pred_tool_calls, expected_tools):
        call_count = len(calls)
        expected_count = len(expected)

        if expected_count == 0:
            scores.append(1.0 if call_count == 0 else 0.5)
        else:
            # 调用数量接近预期 → 高分
            ratio = min(call_count, expected_count) / max(call_count, expected_count, 1)
            scores.append(ratio)

    return sum(scores) / len(scores)


# ================================================================
# 综合评分
# ================================================================


def compute_overall_score(metrics: dict) -> float:
    """根据各项指标计算综合评分.

    Weights:
        - tool_call_accuracy: 25%
        - compliance_rate: 20%
        - hallucination_score: 25%
        - code_success_rate: 10%
        - format_compliance: 10%
        - tool_efficiency: 10%
    """
    weights = {
        "tool_call_accuracy": 0.25,
        "compliance_rate": 0.20,
        "hallucination_score": 0.25,
        "code_success_rate": 0.10,
        "format_compliance": 0.10,
        "tool_efficiency": 0.10,
    }

    overall = 0.0
    for metric, weight in weights.items():
        overall += metrics.get(metric, 0.0) * weight

    return overall
