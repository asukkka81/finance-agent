# model_layer/grpo/reward.py
"""四维加权奖励函数 — GRPO 强化学习的奖励信号.

奖励维度:
    1. format_compliance (20%): 回答格式是否符合规范
    2. tool_scheduling (30%): 工具调用是否合理
    3. code_quality (15%): Python 沙箱代码质量
    4. answer_quality (35%): 答案质量
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RewardBreakdown:
    """各维度奖励明细."""

    format_compliance: float = 0.0
    tool_scheduling: float = 0.0
    code_quality: float = 0.0
    answer_quality: float = 0.0

    def weighted_sum(self, weights: dict[str, float]) -> float:
        """计算加权总分."""
        return (
            weights.get("format_compliance", 0.2) * self.format_compliance
            + weights.get("tool_scheduling", 0.3) * self.tool_scheduling
            + weights.get("code_quality", 0.15) * self.code_quality
            + weights.get("answer_quality", 0.35) * self.answer_quality
        )

    def to_dict(self) -> dict:
        return {
            "format_compliance": self.format_compliance,
            "tool_scheduling": self.tool_scheduling,
            "code_quality": self.code_quality,
            "answer_quality": self.answer_quality,
        }


class RewardCalculator:
    """GRPO 四维奖励计算器.

    奖励设计原则:
        - 每组内计算相对优势 (组内归一化)
        - 每个维度独立评分 (0~1)
        - 最终奖励 = Σ(weight_i × score_i)

    Usage::

        calc = RewardCalculator(config)
        rewards = calc.compute_rewards(rollouts)
        # rollouts = [
        #     {"response": "...", "tool_calls": [...], "code": "..."},
        #     ...
        # ]
    """

    def __init__(self, config):
        self.config = config
        self.weights = config.reward_weights

    # ================================================================
    # 主接口
    # ================================================================

    def compute_rewards(
        self, rollouts: list[dict], prompt: str = ""
    ) -> list[float]:
        """计算一组 rollout 的奖励值.

        Args:
            rollouts: 模型生成的候选回答列表.
            prompt: 原始用户输入.

        Returns:
            每个 rollout 的奖励值列表 (未归一化).
        """
        breakdowns = [self._compute_breakdown(r, prompt) for r in rollouts]
        raw_rewards = [b.weighted_sum(self.weights) for b in breakdowns]

        # 组内相对优势归一化 (GRPO 核心)
        normalized = self._group_normalize(raw_rewards)

        return normalized

    def compute_breakdowns(
        self, rollouts: list[dict], prompt: str = ""
    ) -> list[RewardBreakdown]:
        """计算各维度的详细分数 (用于调试)."""
        return [self._compute_breakdown(r, prompt) for r in rollouts]

    # ================================================================
    # 各维度评分
    # ================================================================

    def _compute_breakdown(
        self, rollout: dict, prompt: str
    ) -> RewardBreakdown:
        """对单个 rollout 计算四维分数."""
        response = rollout.get("response", rollout.get("content", ""))
        tool_calls = rollout.get("tool_calls", [])
        code = rollout.get("code", "")

        return RewardBreakdown(
            format_compliance=self._score_format(response),
            tool_scheduling=self._score_tools(tool_calls, prompt),
            code_quality=self._score_code(code),
            answer_quality=self._score_answer(response, prompt),
        )

    # ----- 格式合规 (20%) -----

    def _score_format(self, response: str) -> float:
        """评估回答格式合规性."""
        score = 1.0

        # 检查风险提示
        risk_keywords = ["风险", "不构成", "仅供参考", "投资需谨慎", "过往业绩"]
        has_risk = any(kw in response for kw in risk_keywords)
        if not has_risk and len(response) > 100:
            score -= 0.3

        # 检查结构完整性
        has_structure = any(marker in response for marker in ["**", "##", "1.", "###"])
        if not has_structure:
            score -= 0.1

        # 检查是否有数据引用
        has_data = bool(re.search(r'[\d.]+%|[\d.]+元|[\d.]+倍', response))
        if not has_data and len(response) > 50:
            score -= 0.1

        return max(0.0, score)

    # ----- 工具调度 (30%) -----

    def _score_tools(
        self, tool_calls: list[dict], prompt: str
    ) -> float:
        """评估工具调用的合理性."""
        if not tool_calls:
            # 如果 prompt 不需要工具，无工具调用是合理的
            return 0.8

        score = 1.0

        # 检查是否有错误工具选择
        tool_names = [tc.get("name", "") for tc in tool_calls]
        valid_tools = {
            "get_stock_price", "search_stocks", "search_knowledge",
            "get_macro_indicator", "get_market_overview",
            "execute_python", "web_search", "web_fetch",
        }
        for name in tool_names:
            if name not in valid_tools:
                score -= 0.2

        # 检查工具数量是否合理
        if len(tool_calls) > 6:
            score -= 0.15  # 过度调用

        # 检查是否有重复调用
        if len(tool_names) != len(set(tool_names)):
            score -= 0.2  # 重复调用同一工具

        # 检查参数完整性
        for tc in tool_calls:
            args = tc.get("arguments", tc.get("input", {}))
            if not args:
                score -= 0.1

        return max(0.0, score)

    # ----- 代码质量 (15%) -----

    def _score_code(self, code: str) -> float:
        """评估 Python 沙箱代码质量."""
        if not code:
            return 0.5  # 不涉及代码，给中间分

        score = 1.0

        # 检查是否使用了预置函数 (sharpe_ratio 等)
        finance_funcs = [
            "sharpe_ratio", "max_drawdown", "sortino_ratio",
            "beta", "alpha", "sma", "ema", "rsi", "macd",
            "portfolio_return", "efficient_frontier",
        ]
        uses_prebuilt = any(fn in code for fn in finance_funcs)
        if not uses_prebuilt:
            score -= 0.1

        # 检查是否有危险代码
        dangerous = [
            "import os", "import sys", "import subprocess",
            "eval(", "exec(", "__import__",
            "open(", "requests.", "urllib.",
        ]
        for pattern in dangerous:
            if pattern in code:
                score -= 0.5
                break

        # 检查是否有 print (输出结果)
        if "print(" not in code:
            score -= 0.1

        # 检查代码长度合理性
        if len(code) > 2000:
            score -= 0.1

        return max(0.0, score)

    # ----- 答案质量 (35%) -----

    def _score_answer(self, response: str, prompt: str) -> float:
        """评估答案质量."""
        score = 1.0

        # 检查是否直接回答了问题
        if len(response) < 20:
            score -= 0.5

        # 检查是否有事实矛盾 (简单检测)
        contradictions = [
            ("上涨", "下跌"),
            ("利好", "利空"),
            ("买入", "卖出"),
        ]
        for pos, neg in contradictions:
            if pos in response and neg in response:
                # 有矛盾表述扣分 (但可能是合理对比)
                if response.count(pos) > 2 and response.count(neg) > 2:
                    score -= 0.05

        # 检查是否包含过度的承诺/确定性语言
        overconfident = ["一定会", "保证", "绝对", "肯定能", "稳赚"]
        for phrase in overconfident:
            if phrase in response:
                score -= 0.15
                break

        # 检查是否包含不恰当的建议
        bad_patterns = ["建议买入", "建议卖出", "all in", "满仓", "梭哈"]
        for pattern in bad_patterns:
            if pattern in response:
                score -= 0.3
                break

        return max(0.0, score)

    # ================================================================
    # 组内相对优势归一化
    # ================================================================

    @staticmethod
    def _group_normalize(rewards: list[float]) -> list[float]:
        """GRPO 组内相对优势归一化.

        advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

        这使得奖励反映组内相对优势，而非绝对质量。
        """
        if not rewards:
            return []

        import numpy as np
        rewards = np.array(rewards, dtype=float)
        mean = np.mean(rewards)
        std = np.std(rewards)

        if std < 1e-8:
            # 所有候选质量相同 → 无优势差异
            return [0.0] * len(rewards)

        advantages = (rewards - mean) / std
        return advantages.tolist()
