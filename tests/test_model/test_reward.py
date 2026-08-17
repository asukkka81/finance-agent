# tests/test_model/test_reward.py
"""GRPO 奖励函数测试."""

import pytest

from model_layer.config import ModelConfig
from model_layer.grpo.reward import RewardCalculator, RewardBreakdown


class TestRewardCalculator:
    """测试四维奖励计算器."""

    @pytest.fixture
    def calc(self):
        config = ModelConfig()
        return RewardCalculator(config)

    def test_format_compliance_with_risk_warning(self, calc):
        """包含风险提示的回答应得高分."""
        response = (
            "根据数据分析，该股票近期表现良好。\n\n"
            "**关键指标**: PE约28倍，ROE维持在30%以上。\n\n"
            "> ⚠️ 免责声明: 以上分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。"
        )
        score = calc._score_format(response)
        assert score >= 0.8, f"Expected >= 0.8, got {score}"

    def test_format_compliance_without_warning(self, calc):
        """缺少风险提示应扣分."""
        response = (
            "该股票非常好，建议立即买入，一定会涨！"
            "机会难得，加仓加仓！"
        )
        score = calc._score_format(response)
        assert score <= 0.9, f"Expected <= 0.9 (missing risk warning), got {score}"

    def test_tool_scheduling_correct(self, calc):
        """正确的工具调用应得高分."""
        tool_calls = [
            {"name": "get_stock_price", "arguments": {"symbol": "600519"}},
            {"name": "search_knowledge", "arguments": {"query": "茅台估值"}},
        ]
        score = calc._score_tools(tool_calls, "茅台估值")
        assert score >= 0.7

    def test_tool_scheduling_unknown_tool(self, calc):
        """错误工具名应扣分."""
        tool_calls = [
            {"name": "unknown_tool_xyz", "arguments": {}},
        ]
        score = calc._score_tools(tool_calls, "test")
        assert score < 0.9

    def test_tool_scheduling_empty(self, calc):
        """无工具调用 (合理的默许)."""
        score = calc._score_tools([], "什么是PE？")
        assert score >= 0.5

    def test_code_quality_safe_code(self, calc):
        """安全的金融代码应得高分."""
        code = (
            "returns = [0.01, -0.02, 0.03]\n"
            "sharpe = sharpe_ratio(returns)\n"
            "print(f\"夏普比率: {sharpe:.4f}\")"
        )
        score = calc._score_code(code)
        assert score >= 0.8

    def test_code_quality_dangerous_import(self, calc):
        """危险导入应得低分."""
        code = (
            "import os\n"
            "os.system('rm -rf /')\n"
            "print('hacked')"
        )
        score = calc._score_code(code)
        assert score < 0.5

    def test_code_quality_empty(self, calc):
        """无代码应得中间分."""
        score = calc._score_code("")
        assert score >= 0.4

    def test_answer_quality_balanced(self, calc):
        """高质量回答应得高分."""
        response = (
            "综合以上数据和知识检索结果，对该股票的分析如下：\n\n"
            "1. **基本面**: PE合理，ROE优秀\n"
            "2. **技术面**: MACD金叉，短期趋势向好\n"
            "3. **风险因素**: 行业政策不确定性\n\n"
            "> ⚠️ 以上分析仅供参考。"
        )
        score = calc._score_answer(response, "分析该股票")
        assert score >= 0.7

    def test_answer_quality_short(self, calc):
        """过短回答应得分较低."""
        score_short = calc._score_answer("涨了", "股票涨了吗")
        score_long = calc._score_answer(
            "根据最新数据分析，该股票近期呈现温和上涨趋势... > ⚠️ 仅供参考",
            "股票涨了吗",
        )
        # 长回答应比短回答得分高
        assert score_long > score_short, f"short={score_short}, long={score_long}"

    def test_group_normalize(self, calc):
        """组内归一化: 相同值 → 0优势."""
        rewards = [0.5, 0.5, 0.5, 0.5]
        normalized = calc._group_normalize(rewards)
        assert all(abs(a) < 1e-6 for a in normalized)

    def test_group_normalize_varied(self, calc):
        """组内归一化: 不同值 → 差分优势."""
        rewards = [0.2, 0.5, 0.8]
        normalized = calc._group_normalize(rewards)
        assert normalized[2] > 0  # 最高分 → 正优势
        assert normalized[0] < 0  # 最低分 → 负优势

    def test_compute_breakdown(self, calc):
        """完整四维度分解."""
        rollout = {
            "response": (
                "分析结果：该股票PE合理。\n\n"
                "> ⚠️ 仅供参考，不构成投资建议。"
            ),
            "tool_calls": [
                {"name": "get_stock_price", "arguments": {"symbol": "600519"}},
            ],
            "code": "sharpe = sharpe_ratio([0.01, 0.02])\nprint(sharpe)",
        }
        breakdown = calc._compute_breakdown(rollout, "分析股票")
        assert isinstance(breakdown, RewardBreakdown)
        assert 0 <= breakdown.format_compliance <= 1
        assert 0 <= breakdown.tool_scheduling <= 1
        assert 0 <= breakdown.code_quality <= 1
        assert 0 <= breakdown.answer_quality <= 1

    def test_compute_rewards(self, calc):
        """完整奖励计算."""
        rollouts = [
            {
                "response": "高质量分析... > ⚠️ 仅供参考",
                "tool_calls": [{"name": "get_stock_price", "arguments": {"symbol": "600519"}}],
                "code": "sharpe = sharpe_ratio(returns)\nprint(sharpe)",
            },
            {
                "response": "短回答",
                "tool_calls": [],
                "code": "",
            },
        ]
        rewards = calc.compute_rewards(rollouts, "分析股票")
        assert len(rewards) == 2
        # 高质量应得高奖励
        assert rewards[0] > rewards[1]
