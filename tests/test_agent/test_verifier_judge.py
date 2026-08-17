# tests/test_agent/test_verifier_judge.py
"""LLM Judge 校验 & 低分重答循环 & run_stream 回归测试."""

import json

import pytest

from agent_layer.config import AgentConfig
from agent_layer.core.verifier import Verifier, Verdict
from agent_layer.llm.base import BaseLLMClient, LLMResponse
from agent_layer.mcp.registry import ToolRegistry
from agent_layer.orchestrator import AgentOrchestrator


# ================================================================
# 测试工具
# ================================================================

TOOL_RESULTS = [
    {"tool_name": "get_stock_price", "success": True,
     "data": {"summary": {"symbol": "600519", "latest_close": 1850.50}}},
]


def _judge_response(acc: float, logic: float,
                    issues=None, suggestions=None) -> LLMResponse:
    """构造 judge JSON 响应."""
    return LLMResponse(content=json.dumps({
        "accuracy_score": acc,
        "logic_score": logic,
        "issues": issues or [],
        "suggestions": suggestions or [],
    }, ensure_ascii=False), finish_reason="stop")


class ScriptedLLMClient(BaseLLMClient):
    """按脚本返回预定义响应的 LLM 客户端.

    chat() 每次调用依次弹出脚本中的下一条响应
    (校验 judge 与重答共用同一脚本, 按调用顺序).
    """

    def __init__(self, chat_script: list[LLMResponse]):
        self.chat_script = list(chat_script)
        self._idx = 0
        self.chat_calls = 0

    def chat(self, messages, system_prompt=None, **kwargs) -> LLMResponse:
        resp = self.chat_script[min(self._idx, len(self.chat_script) - 1)]
        self._idx += 1
        self.chat_calls += 1
        return resp

    def chat_with_tools(self, messages, tools, system_prompt=None, **kwargs) -> LLMResponse:
        return LLMResponse(content="", tool_calls=[], finish_reason="stop")

    def chat_with_tool_results(self, messages, tool_results, tools,
                               system_prompt=None, **kwargs) -> LLMResponse:
        return LLMResponse(content="", finish_reason="stop")


# ================================================================
# LLM Judge
# ================================================================

class TestLLMJudge:
    """校验器接入 LLM judge 的测试."""

    def test_judge_scores_used(self):
        """judge 返回的分数应被 accuracy/logic 维度采用."""
        client = ScriptedLLMClient([_judge_response(0.9, 0.85)])
        verifier = Verifier(use_llm=True, llm_client=client)

        result = verifier.verify_final_answer(
            "茅台股价", "茅台最新收盘价 1850.50 元。", TOOL_RESULTS, "stock_price",
        )
        dims = {d.dimension: d for d in result.dimensions}
        assert dims["accuracy"].score == 0.9
        assert dims["logic"].score == 0.85
        assert dims["accuracy"].verdict == Verdict.PASS

    def test_judge_low_score_marks_fail(self):
        """judge 低分 → FAIL, issues 透传."""
        client = ScriptedLLMClient([
            _judge_response(0.3, 0.4, issues=["数字与工具结果不一致"]),
        ])
        verifier = Verifier(use_llm=True, llm_client=client)

        result = verifier.verify_final_answer(
            "茅台股价", "茅台最新收盘价 2000 元。", TOOL_RESULTS, "stock_price",
        )
        dims = {d.dimension: d for d in result.dimensions}
        assert dims["accuracy"].verdict == Verdict.FAIL
        assert dims["accuracy"].issues == ["数字与工具结果不一致"]
        assert result.passed is False

    def test_bad_json_falls_back_to_rules(self):
        """judge 返回非 JSON → 回退规则校验, 不抛异常."""
        client = ScriptedLLMClient([LLMResponse(content="这不是 JSON", finish_reason="stop")])
        verifier = Verifier(use_llm=True, llm_client=client)

        result = verifier.verify_final_answer(
            "茅台股价", "茅台最新收盘价 1850.50 元。", TOOL_RESULTS, "stock_price",
        )
        dims = {d.dimension: d for d in result.dimensions}
        assert dims["accuracy"].score == 0.8  # 规则回退值

    def test_judge_error_falls_back(self):
        """judge 调用抛异常 → 回退规则校验, 不抛异常."""
        class FailingClient(ScriptedLLMClient):
            def chat(self, messages, system_prompt=None, **kwargs):
                raise RuntimeError("LLM down")

        verifier = Verifier(use_llm=True, llm_client=FailingClient([]))
        result = verifier.verify_final_answer(
            "茅台股价", "茅台最新收盘价 1850.50 元。", TOOL_RESULTS, "stock_price",
        )
        assert result.overall_score > 0

    def test_judge_disabled_without_llm(self):
        """无 LLM 时不调用 judge."""
        verifier = Verifier()
        result = verifier.verify_final_answer(
            "茅台股价", "答案", TOOL_RESULTS, "stock_price",
        )
        dims = {d.dimension: d for d in result.dimensions}
        assert dims["accuracy"].score == 0.8  # 规则回退
        assert dims["logic"].score == 0.65   # 规则回退


# ================================================================
# 低分重答循环
# ================================================================

class TestRegenerationLoop:
    """分数低于阈值 → 重答 → 仍低 → 降级 的循环测试."""

    def _orchestrator(self, script) -> tuple[AgentOrchestrator, ScriptedLLMClient]:
        client = ScriptedLLMClient(script)
        config = AgentConfig(
            max_tool_rounds=2,
            min_verification_score=0.7,
            max_regeneration_rounds=2,
        )
        return AgentOrchestrator(config=config, registry=None, llm_client=client), client

    def test_low_score_triggers_regeneration(self):
        """低分 → 重答一次 → 达标, 返回重答后的答案."""
        script = [
            _judge_response(0.4, 0.4, issues=["数字与工具结果不一致"]),   # 首次校验
            LLMResponse(content="修正后的回答: 茅台最新收盘价 1850.50 元。"),  # 重答
            _judge_response(0.9, 0.85),                                   # 重答后校验
        ]
        orch, client = self._orchestrator(script)

        final, verification = orch._verify_and_regenerate(
            query="茅台股价",
            answer="茅台最新收盘价 2000 元。",
            tool_results=TOOL_RESULTS,
            intent_type="stock_price",
        )
        assert final == "修正后的回答: 茅台最新收盘价 1850.50 元。"
        assert verification.overall_score >= 0.7
        assert client.chat_calls == 3

    def test_improving_but_still_low_degrades(self):
        """每次重答都有改善但最终仍低于阈值 → 用完重答次数后降级."""
        script = [
            _judge_response(0.3, 0.3, issues=["数字与工具结果不一致"]),
            LLMResponse(content="改善一点的回答。"),
            _judge_response(0.35, 0.35, issues=["数字与工具结果不一致"]),
            LLMResponse(content="再改善一点的回答。"),
            _judge_response(0.4, 0.4, issues=["数字与工具结果不一致"]),
        ]
        orch, client = self._orchestrator(script)

        final, verification = orch._verify_and_regenerate(
            query="茅台股价",
            answer="茅台最新收盘价 2000 元。",
            tool_results=TOOL_RESULTS,
            intent_type="stock_price",
        )
        # 2 次重答全部用掉, 分数仍 < 0.7 → 降级追加质量校验提示
        assert client.chat_calls == 5
        assert final == "再改善一点的回答。" or final.startswith("再改善一点的回答。")
        assert "质量校验提示" in final
        assert verification.overall_score < 0.7

    def test_no_improvement_stops_early(self):
        """重答无改善 → 提前停止, 不再浪费 LLM 调用."""
        script = [
            _judge_response(0.4, 0.4, issues=["问题"]),
            LLMResponse(content="一样差的回答。"),
            _judge_response(0.4, 0.4, issues=["问题"]),
        ]
        orch, client = self._orchestrator(script)

        final, verification = orch._verify_and_regenerate(
            query="茅台股价", answer="差的回答。",
            tool_results=TOOL_RESULTS, intent_type="stock_price",
        )
        assert client.chat_calls == 3  # 1 次校验 + 1 次重答 + 1 次重答后校验
        assert "质量校验提示" in final

    def test_score_above_threshold_no_regeneration(self):
        """分数达标 → 直接返回原答案, 不重答."""
        script = [_judge_response(0.9, 0.9)]
        orch, client = self._orchestrator(script)

        final, verification = orch._verify_and_regenerate(
            query="茅台股价", answer="好的回答。",
            tool_results=TOOL_RESULTS, intent_type="stock_price",
        )
        assert final == "好的回答。"
        assert client.chat_calls == 1


class TestCompletenessRules:
    """规则模式完整性校验: 实体覆盖 + 需求清单 + 数据类型."""

    PRICE_TOOL = [
        {"tool_name": "get_stock_price", "success": True,
         "data": {"summary": {"symbol": "600519", "latest_close": 1850.50}}},
    ]

    def test_full_coverage_passes(self):
        """实体 + 需求 + 数据全部覆盖 → 高分 PASS."""
        verifier = Verifier()
        result = verifier.verify_final_answer(
            "对比一下茅台和五粮液，谁更值得投资？",
            "茅台 ROE 30%，PE 28 倍；五粮液 ROE 25%，PE 22 倍。"
            "相比而言五粮液估值更低，更值得关注。投资有风险。",
            self.PRICE_TOOL,
            "stock_price",
        )
        dims = {d.dimension: d for d in result.dimensions}
        comp = dims["completeness"]
        assert comp.verdict == Verdict.PASS
        assert comp.score == 1.0

    def test_missing_entity_detected(self):
        """查询提到两只股票但回答只提一只 → 实体缺失被检出."""
        verifier = Verifier()
        result = verifier.verify_final_answer(
            "对比一下茅台和五粮液，谁更值得投资？",
            "茅台 ROE 30%，估值合理，值得关注。",  # 没提五粮液
            self.PRICE_TOOL,
            "stock_price",
        )
        dims = {d.dimension: d for d in result.dimensions}
        comp = dims["completeness"]
        assert "五粮液" in " ".join(comp.issues)
        assert comp.score < 0.8  # 实体覆盖率 50%

    def test_missing_requirement_detected(self):
        """问估值+行业地位, 回答只讲估值 → 需求缺失被检出."""
        verifier = Verifier()
        result = verifier.verify_final_answer(
            "帮我分析一下茅台的估值水平和行业地位",
            "茅台 PE 28 倍，估值处于历史中位。",  # 没讲行业地位
            self.PRICE_TOOL,
            "stock_price",
        )
        dims = {d.dimension: d for d in result.dimensions}
        comp = dims["completeness"]
        assert "行业地位" in " ".join(comp.issues)
        assert comp.verdict == Verdict.WEAK

    def test_missing_data_triggers_needs_more(self):
        """数据类型缺失 → NEEDS_MORE + 补充调用建议 (原行为保留)."""
        verifier = Verifier()
        result = verifier.verify_after_tool_call(
            "茅台股价", [], "stock_price",
        )
        assert result.needs_more_info
        assert result.suggested_tool_calls == [
            {"tool": "get_stock_price", "reason": "需要获取股票行情数据"},
        ]

    def test_indicator_abbrevs_not_treated_as_entities(self):
        """PE/CPI 等指标缩写不应被误提取为股票实体."""
        from agent_layer.core.verifier import extract_query_entities

        entities = extract_query_entities("用PE和CPI分析茅台估值")
        codes = {e["code"] for e in entities}
        assert "PE" not in codes
        assert "CPI" not in codes
        assert "600519" in codes  # 茅台被正确提取

    def test_judge_completeness_used(self):
        """judge 返回 completeness_score 时直接采用."""
        import json as _json

        client = ScriptedLLMClient([LLMResponse(content=_json.dumps({
            "completeness_score": 0.3,
            "accuracy_score": 0.9,
            "logic_score": 0.9,
            "missing_items": ["缺少行业地位分析"],
        }, ensure_ascii=False), finish_reason="stop")])
        verifier = Verifier(use_llm=True, llm_client=client)

        result = verifier.verify_final_answer(
            "帮我分析一下茅台的估值水平和行业地位",
            "茅台 PE 28 倍。",
            TOOL_RESULTS,
            "stock_price",
        )
        dims = {d.dimension: d for d in result.dimensions}
        assert dims["completeness"].score == 0.3
        assert dims["completeness"].issues == ["缺少行业地位分析"]
        assert dims["completeness"].verdict == Verdict.FAIL


# ================================================================
# run_stream 回归
# ================================================================

class TestRunStream:
    """run_stream 回归测试."""

    def test_tool_executed_once(self):
        """工具在一次流式执行中只应执行一次 (双执行 bug 回归)."""
        reg = ToolRegistry()
        calls = {"n": 0}

        @reg.register(
            "count_tool",
            "计数工具",
            parameters={"type": "object", "properties": {}},
        )
        def count_tool(**kwargs) -> dict:
            calls["n"] += 1
            return {"count": calls["n"]}

        class OneShotClient(BaseLLMClient):
            def chat(self, messages, system_prompt=None, **kwargs) -> LLMResponse:
                return LLMResponse(content="好的回答。", finish_reason="stop")

            def chat_with_tools(self, messages, tools, system_prompt=None, **kwargs) -> LLMResponse:
                return LLMResponse(
                    content="",
                    tool_calls=[{"id": "c1", "name": "count_tool", "input": {}}],
                    finish_reason="tool_use",
                )

            def chat_with_tool_results(self, messages, tool_results, tools,
                                       system_prompt=None, **kwargs) -> LLMResponse:
                return LLMResponse(content="基于结果回答。", finish_reason="stop")

        orch = AgentOrchestrator(
            config=AgentConfig(max_tool_rounds=2),
            registry=reg,
            llm_client=OneShotClient(),
        )
        events = list(orch.run_stream("测试"))

        assert calls["n"] == 1  # 只执行一次
        answer_events = [e for e in events if e["type"] == "answer"]
        assert len(answer_events) == 1
        assert "verification" in answer_events[0]  # 最终校验已接入
