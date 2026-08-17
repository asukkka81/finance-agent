# model_layer/evaluation/benchmark.py
"""金融 Agent Benchmark — 标准化评测数据集.

覆盖场景:
    - stock_qa: 股票问答 (行情查询、指标计算)
    - knowledge_qa: 金融知识问答
    - macro_analysis: 宏观经济分析
    - investment_advice: 投资建议 (需合规检查)
    - tool_usage: 工具调用链
    - sandbox_code: Python 代码执行
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    """一条 Benchmarks 样本."""

    query: str
    category: str
    expected_tools: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)
    expected_compliance: bool = False  # 是否需要合规提示
    ground_truth: str = ""


class FinanceBenchmark:
    """金融 Agent 评测基准.

    内置标准评测集，覆盖 6 大场景。
    """

    def __init__(self):
        self.samples: list[BenchmarkSample] = []
        self._build_default_benchmark()

    # ================================================================
    # 内置评测数据
    # ================================================================

    def _build_default_benchmark(self):
        """构建内置评测数据集 (精选样本)."""
        self.samples = [
            # --- 股票行情 ---
            BenchmarkSample(
                query="贵州茅台（600519）最近一个月的股价走势如何？",
                category="stock_qa",
                expected_tools=["get_stock_price"],
                expected_facts=["600519", "股价", "收盘"],
            ),
            BenchmarkSample(
                query="帮我分析Apple最新的季度表现",
                category="stock_qa",
                expected_tools=["search_stocks", "get_stock_price"],
                expected_facts=["AAPL", "Apple"],
            ),

            # --- 知识问答 ---
            BenchmarkSample(
                query="什么是夏普比率？如何用它来评估基金？",
                category="knowledge_qa",
                expected_tools=["search_knowledge"],
                expected_facts=["夏普比率", "超额回报", "风险调整"],
                expected_compliance=False,
            ),
            BenchmarkSample(
                query="PE和PB有什么区别？分别适用于什么场景？",
                category="knowledge_qa",
                expected_tools=["search_knowledge"],
                expected_facts=["市盈率", "市净率", "估值"],
            ),

            # --- 宏观分析 ---
            BenchmarkSample(
                query="最新的CPI和PMI数据说明了什么经济趋势？",
                category="macro_analysis",
                expected_tools=["get_macro_indicator", "search_knowledge"],
                expected_facts=["CPI", "PMI", "宏观"],
                expected_compliance=False,
            ),

            # --- 投资建议 ---
            BenchmarkSample(
                query="我是一名风险偏好较低的投资者，应该如何配置资产？",
                category="investment_advice",
                expected_tools=["search_knowledge"],
                expected_facts=["资产配置", "风险", "组合"],
                expected_compliance=True,
            ),

            # --- 代码执行 ---
            BenchmarkSample(
                query="帮我计算如果投资组合每天收益0.1%，一年的夏普比率是多少",
                category="sandbox_code",
                expected_tools=["execute_python"],
                expected_facts=["sharpe_ratio", "0.001", "252"],
            ),

            # --- 工具链 ---
            BenchmarkSample(
                query="搜索新能源相关的A股，然后分析宁德时代的估值",
                category="tool_usage",
                expected_tools=["search_stocks", "get_stock_price", "search_knowledge"],
                expected_facts=["新能源", "宁德时代", "300750"],
            ),
        ]

    # ================================================================
    # 评测执行
    # ================================================================

    def evaluate(
        self,
        generate_fn,
        verbose: bool = False,
    ) -> dict:
        """运行 Benchmark 评测.

        Args:
            generate_fn: 生成函数 (prompt) → {"response": str, "tool_calls": [...]}.
            verbose: 是否打印详细结果.

        Returns:
            评测结果字典.
        """
        from model_layer.evaluation.metrics import compute_all_metrics, compute_overall_score

        predictions = []
        references = []

        for sample in self.samples:
            result = generate_fn(sample.query)

            predictions.append({
                "response": result.get("response", result.get("text", "")),
                "tool_calls": result.get("tool_calls", []),
                "code_result": result.get("code_result", {}),
            })

            references.append({
                "expected_tools": sample.expected_tools,
                "expected_facts": sample.expected_facts,
                "expected_compliance": sample.expected_compliance,
            })

            if verbose:
                logger.info(
                    "Q: %s\nExpected tools: %s\nGot tools: %s\n",
                    sample.query[:60],
                    sample.expected_tools,
                    [tc.get("name", "") for tc in result.get("tool_calls", [])],
                )

        metrics = compute_all_metrics(predictions, references)
        overall = compute_overall_score(metrics)

        results = {
            "total_samples": len(self.samples),
            "metrics": metrics,
            "overall_score": overall,
            "by_category": self._compute_by_category(predictions),
        }

        logger.info("Benchmark complete: overall=%.3f", overall)
        return results

    def _compute_by_category(self, predictions: list[dict]) -> dict:
        """按类别统计分数."""
        categories = {}
        for sample, pred in zip(self.samples, predictions):
            cat = sample.category
            if cat not in categories:
                categories[cat] = {"total": 0, "score": 0.0}

            categories[cat]["total"] += 1
            # 简单评分: 预期工具是否被调用
            pred_tools = {tc.get("name", "") for tc in pred.get("tool_calls", [])}
            expected = set(sample.expected_tools)
            if expected:
                match = len(pred_tools & expected) / len(expected)
            else:
                match = 1.0 if not pred_tools else 0.5
            categories[cat]["score"] += match

        for cat in categories:
            categories[cat]["score"] /= categories[cat]["total"]

        return categories

    # ================================================================
    # 加载自定义评测集
    # ================================================================

    def load_from_file(self, path: str):
        """从 JSONL 文件加载自定义评测集.

        每行格式: {"query": ..., "category": ..., "expected_tools": [...]}
        """
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.samples.append(BenchmarkSample(
                    query=record["query"],
                    category=record.get("category", "custom"),
                    expected_tools=record.get("expected_tools", []),
                    expected_facts=record.get("expected_facts", []),
                    expected_compliance=record.get("expected_compliance", False),
                ))

        logger.info("Loaded %d benchmark samples from %s", len(self.samples), path)

    def __len__(self) -> int:
        return len(self.samples)
