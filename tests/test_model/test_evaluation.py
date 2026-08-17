# tests/test_model/test_evaluation.py
"""评估指标 & Benchmark 测试."""

import pytest

from model_layer.evaluation.metrics import (
    compute_compliance_rate,
    compute_tool_accuracy,
    compute_format_compliance,
    compute_hallucination_score,
    compute_code_success_rate,
    compute_overall_score,
)
from model_layer.evaluation.benchmark import FinanceBenchmark, BenchmarkSample


class TestMetrics:
    """测试各项评估指标."""

    def test_tool_accuracy_exact_match(self):
        preds = [{"tool_calls": [{"name": "get_stock_price"}]}]
        refs = [{"expected_tools": ["get_stock_price"]}]
        acc = compute_tool_accuracy(preds, refs)
        assert acc == 1.0

    def test_tool_accuracy_partial_match(self):
        preds = [{"tool_calls": [{"name": "get_stock_price"}, {"name": "extra_tool"}]}]
        refs = [{"expected_tools": ["get_stock_price"]}]
        acc = compute_tool_accuracy(preds, refs)
        assert acc == 1.0  # 预期工具都在

    def test_tool_accuracy_no_match(self):
        preds = [{"tool_calls": [{"name": "wrong_tool"}]}]
        refs = [{"expected_tools": ["get_stock_price"]}]
        acc = compute_tool_accuracy(preds, refs)
        assert acc == 0.0

    def test_compliance_rate(self):
        responses = [
            "分析内容...\n> ⚠️ 仅供参考，不构成投资建议。投资有风险，入市需谨慎。",
            "买入！一定会涨！",
        ]
        rate = compute_compliance_rate(responses)
        assert rate > 0.5  # 第一个合规

    def test_hallucination_score(self):
        responses = ["苹果公司（Apple）的股票代码是AAPL，最新财报显示营收增长。"]
        facts = [["AAPL", "Apple", "营收"]]
        score = compute_hallucination_score(responses, facts)
        assert score > 0.5

    def test_code_success_rate(self):
        results = [
            {"success": True},
            {"success": True},
            {"success": False},
        ]
        rate = compute_code_success_rate(results)
        assert rate == 2 / 3

    def test_format_compliance(self):
        responses = [
            "**标题**\n内容有**加粗**和数字123%",
            "没有格式的纯文本回答",
        ]
        score = compute_format_compliance(responses)
        assert 0 < score < 1

    def test_overall_score(self):
        metrics = {
            "tool_call_accuracy": 0.90,
            "compliance_rate": 0.85,
            "hallucination_score": 0.80,
            "code_success_rate": 0.95,
            "format_compliance": 0.75,
            "tool_efficiency": 0.70,
        }
        overall = compute_overall_score(metrics)
        assert 0.7 < overall < 0.95


class TestBenchmark:
    """测试 Benchmark."""

    def test_default_benchmark(self):
        bm = FinanceBenchmark()
        assert len(bm) > 0
        # 所有样本应有 query
        for s in bm.samples:
            assert s.query
            assert s.category

    def test_benchmark_evaluate(self):
        bm = FinanceBenchmark()

        def mock_generate(prompt):
            return {
                "response": f"Mock analysis for: {prompt[:30]}... > ⚠️ 仅供参考",
                "tool_calls": [
                    {"name": "search_knowledge", "arguments": {"query": prompt}},
                ],
            }

        result = bm.evaluate(mock_generate, verbose=False)
        assert "metrics" in result
        assert "overall_score" in result
        assert "by_category" in result
        assert result["total_samples"] == len(bm)

    def test_benchmark_categories(self):
        bm = FinanceBenchmark()
        categories = {s.category for s in bm.samples}
        assert "stock_qa" in categories
        assert "knowledge_qa" in categories
        assert "macro_analysis" in categories
        assert "investment_advice" in categories
        assert "sandbox_code" in categories
        assert "tool_usage" in categories
