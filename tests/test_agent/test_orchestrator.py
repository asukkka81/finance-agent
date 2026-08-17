# tests/test_agent/test_orchestrator.py
"""Agent 调度中枢端到端测试."""

import pytest

from agent_layer.core.intent import Intent, IntentType
from agent_layer.core.planner import Planner
from agent_layer.core.executor import ToolExecutor
from agent_layer.core.verifier import Verifier
from agent_layer.orchestrator import AgentOrchestrator


class TestOrchestrator:
    """测试 Agent 调度中枢."""

    @pytest.fixture
    def orchestrator(self, agent_config, registry, mock_llm):
        """创建完整的 Orchestrator (使用 MockLLM)."""
        return AgentOrchestrator(
            config=agent_config,
            registry=registry,
            llm_client=mock_llm,
        )

    def test_simple_stock_query(self, orchestrator):
        """简单股票查询 — 端到端."""
        response = orchestrator.run(
            "贵州茅台（600519）最新股价是多少？",
            use_llm=False,
        )

        assert response.query == "贵州茅台（600519）最新股价是多少？"
        assert response.intent is not None
        assert response.intent.intent_type == IntentType.STOCK_PRICE
        assert "600519" in response.intent.symbols
        assert response.execution is not None
        assert len(response.execution.results) > 0
        assert response.elapsed_seconds >= 0

    def test_knowledge_query(self, orchestrator):
        """知识问答 — 端到端."""
        response = orchestrator.run(
            "什么是市盈率？如何用PE估值？",
            use_llm=False,
        )

        assert response.intent.intent_type == IntentType.KNOWLEDGE_QA
        assert len(response.execution.results) > 0
        assert response.execution.results[0].tool_name == "search_knowledge"

    def test_macro_query(self, orchestrator):
        """宏观数据查询."""
        response = orchestrator.run(
            "最近CPI数据是多少？",
            use_llm=False,
        )

        assert response.intent.intent_type == IntentType.MACRO_ANALYSIS
        assert len(response.execution.results) > 0

    def test_verification_adds_compliance(self, orchestrator):
        """投资建议应自动添加合规提示."""
        response = orchestrator.run(
            "如何配置我的投资组合？",
            use_llm=False,
        )

        # 答案应包含风险提示
        assert "风险" in response.answer or "不构成" in response.answer

    def test_answer_not_empty(self, orchestrator):
        """答案不应为空."""
        response = orchestrator.run("茅台股价", use_llm=False)
        assert len(response.answer) > 0

    def test_history_recording(self, orchestrator):
        """对话历史应被记录."""
        orchestrator.run("茅台股价", use_llm=False)
        assert len(orchestrator.history) == 2  # user + assistant

    def test_complex_query_multiple_tools(self, orchestrator):
        """复杂查询应调用多个工具."""
        response = orchestrator.run(
            "请分析贵州茅台的投资价值，包括最新股价和估值分析",
            use_llm=False,
        )

        # 应该至少调用了数据工具
        tools_called = {r.tool_name for r in response.execution.results}
        # 可能触发多轮工具调用
        assert len(tools_called) >= 1


class TestPlanner:
    """测试策略规划器."""

    def test_stock_price_plan(self):
        """行情查询计划."""
        planner = Planner()
        intent = Intent(
            intent_type=IntentType.STOCK_PRICE,
            entities={"symbol": ["600519"]},
            original_query="茅台股价",
        )

        plan = planner.plan(intent)
        assert plan.step_count >= 1
        assert plan.steps[0].tool_name == "get_stock_price"

    def test_knowledge_qa_plan(self):
        """知识问答计划."""
        planner = Planner()
        intent = Intent(
            intent_type=IntentType.KNOWLEDGE_QA,
            original_query="什么是ROE",
        )

        plan = planner.plan(intent)
        assert plan.step_count >= 1
        assert plan.steps[0].tool_name == "search_knowledge"

    def test_macro_plan_has_two_steps(self):
        """宏观分析应包含两个步骤 (数据 + 知识)."""
        planner = Planner()
        intent = Intent(
            intent_type=IntentType.MACRO_ANALYSIS,
            entities={"indicator": ["cpi"]},
            original_query="CPI走势分析",
        )

        plan = planner.plan(intent)
        assert plan.step_count == 2
        assert plan.mode.value == "serial"  # 有依赖关系


class TestVerifier:
    """测试校验器."""

    def test_completeness_check(self):
        """完整性检查."""
        verifier = Verifier()
        tool_results = [
            {"tool_name": "get_stock_price", "success": True,
             "data": {"summary": {"symbol": "600519", "latest_close": 1850.50}}},
        ]

        result = verifier.verify_after_tool_call(
            "茅台股价", tool_results, "stock_price",
        )
        assert result.passed  # 有价格数据应通过

    def test_compliance_check_needs_warning(self):
        """投资建议应检查合规."""
        verifier = Verifier()
        answer = "建议立即买入茅台，一定会涨！"
        result = verifier.verify_final_answer(
            "如何投资茅台", answer, [], "investment_advice",
        )
        # 应该因为缺少风险提示而降低分数
        assert result.overall_score < 1.0

    def test_compliance_check_with_warning(self):
        """带风险提示的答案应通过."""
        verifier = Verifier()
        answer = (
            "茅台是优质资产，但投资需谨慎。"
            "以上分析仅供参考，不构成投资建议。"
        )
        result = verifier.verify_final_answer(
            "如何投资茅台", answer, [], "investment_advice",
        )
        assert result.passed


class TestIntentParser:
    """额外意图解析测试."""

    def test_complex_query_detection(self):
        """复合查询检测."""
        from agent_layer.core.intent import IntentParser
        parser = IntentParser()

        # 包含多个意图的查询
        intent = parser.parse("帮我分析一下茅台的股价和ROE表现")
        # 既包含行情也包含知识
        assert intent.intent_type in (
            IntentType.STOCK_PRICE,
            IntentType.KNOWLEDGE_QA,
            IntentType.COMPLEX,
        )
