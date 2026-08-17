# agent_layer/orchestrator.py
"""Agent 调度中枢 — 意图理解 → 策略规划 → 执行调度 → 多轮校验.

这是 Agent 层的顶层入口，连接 MCP 工具系统、LLM 客户端和校验器。
"""

import logging
from datetime import datetime
from typing import Optional

from agent_layer.config import AgentConfig
from agent_layer.core.executor import ExecutionResult, ToolExecutor
from agent_layer.core.intent import Intent, IntentParser, IntentType
from agent_layer.core.planner import Plan, Planner
from agent_layer.core.verifier import (
    VerificationResult,
    Verifier,
    format_tool_results_text,
)
from agent_layer.llm.base import BaseLLMClient, LLMResponse
from agent_layer.mcp.registry import ToolRegistry
from agent_layer.mcp.types import ToolCall
from agent_layer.prompts.templates import (
    FINANCIAL_ADVISOR_PROMPT,
    REGENERATION_PROMPT,
)

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Agent 调度中枢.

    完整的调度流程:
        1. 意图理解: 解析用户查询 → Intent
        2. 策略规划: Intent → Plan (工具调用计划)
        3. 执行调度: 按计划调用工具 (串行/并行)
        4. 中间校验: 检查信息完整性 → 不足则补充查询
        5. 答案生成: 工具结果 + LLM → 最终答案
        6. 最终校验: 四维度打分 → 不合规则修正

    Usage::

        config = AgentConfig()
        registry = ToolRegistry()
        # ... register tools ...

        agent = AgentOrchestrator(config, registry, llm_client)
        response = agent.run("请分析贵州茅台的最新估值水平")
        print(response.answer)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        registry: Optional[ToolRegistry] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.config = config or AgentConfig()
        self.registry = registry or ToolRegistry()
        self.llm_client = llm_client

        # 初始化各组件
        self.intent_parser = IntentParser()
        self.planner = Planner(
            use_llm=bool(llm_client),
            llm_client=llm_client,
        )
        self.executor = ToolExecutor(self.registry)
        self.verifier = Verifier(
            use_llm=bool(llm_client),
            llm_client=llm_client,
        )

        # 对话历史
        self._conversation_history: list[dict] = []

        logger.info(
            "AgentOrchestrator initialized: %d tools, llm=%s",
            self.registry.tool_count,
            self.llm_client is not None,
        )

    # ================================================================
    # 主运行方法
    # ================================================================

    def run(self, query: str, use_llm: bool = True) -> "AgentResponse":
        """执行一次完整的 Agent 查询-响应循环.

        Args:
            query: 用户自然语言查询.
            use_llm: 是否使用 LLM 生成答案 (False 时仅做工具调用).

        Returns:
            AgentResponse 包含答案、工具调用记录、校验结果.
        """
        t_start = datetime.now()

        # ---- Phase 1: 意图理解 ----
        intent = self.intent_parser.parse(query)
        self._add_to_history("user", query)

        if use_llm and self.llm_client:
            # LLM 主导模式: LLM 决定工具调用
            return self._run_with_llm(query, intent)

        # ---- Phase 2: 策略规划 ----
        plan = self.planner.plan(intent)

        # ---- Phase 3: 执行调度 ----
        exec_result = self._execute_with_verification(plan, query, intent)

        # ---- Phase 4: 答案生成 (rule-based) ----
        answer = self._generate_answer_rule_based(
            query, exec_result, intent,
        )

        # ---- Phase 5: 最终校验 ----
        verification = self.verifier.verify_final_answer(
            query=query,
            answer=answer,
            tool_results=[
                r.to_dict() for r in exec_result.results
            ],
            intent_type=intent.intent_type.value,
        )

        # 如果校验不通过且还有余量，尝试修正
        if not verification.passed and exec_result.success:
            answer = self._apply_compliance_fix(answer, verification)

        self._add_to_history("assistant", answer)

        elapsed = (datetime.now() - t_start).total_seconds()

        response = AgentResponse(
            query=query,
            answer=answer,
            intent=intent,
            plan=plan,
            execution=exec_result,
            verification=verification,
            elapsed_seconds=elapsed,
        )

        logger.info(
            "Agent run complete: intent=%s, tools=%d, time=%.1fs",
            intent.intent_type.value,
            len(exec_result.results),
            elapsed,
        )
        return response

    def run_stream(self, query: str, use_llm: bool = True):
        """流式执行 Agent 查询-响应循环，逐步 yield 思维链状态.

        每一步工具调用或获得答案，都通过 generator yield 出来，
        让前端能实时显示思考过程。
        最终答案会经过最终校验 (与 run 路径一致), 低分时触发重新生成。

        Args:
            query: 用户自然语言查询.

        Yields:
            dict: {"type": "thinking"|"tool_call"|"tool_result"|"answer"|"error", ...}
        """
        if not self.llm_client:
            yield {"type": "error", "message": "LLM 客户端未初始化"}
            return

        t_start = datetime.now()
        tools = self.registry.get_tool_schemas()
        system_prompt = self._get_system_prompt()
        intent = self.intent_parser.parse(query)
        self._add_to_history("user", query)

        yield {
            "type": "thinking",
            "content": f"意图识别: {intent.intent_type.value}，准备调用工具…",
            "round": 0,
        }

        messages = [{"role": "user", "content": query}]
        all_results = []  # 累计全部工具执行结果 (用于最终校验)
        max_rounds = self.config.max_tool_rounds

        for round_num in range(1, max_rounds + 1):
            # 调用 LLM
            yield {
                "type": "thinking",
                "content": f"正在与 LLM 通信 (第 {round_num}/{max_rounds} 轮)…",
                "round": round_num,
            }

            if round_num == 1:
                resp = self.llm_client.chat_with_tools(
                    messages, tools=tools, system_prompt=system_prompt,
                )
            else:
                resp = self.llm_client.chat_with_tool_results(
                    messages, tool_results=last_results, tools=tools,
                    system_prompt=system_prompt,
                )

            # LLM 返回了 tool_calls → 执行
            if resp.has_tool_calls:
                thinking = resp.content.strip() if resp.content else ""
                if thinking:
                    yield {
                        "type": "thinking",
                        "content": thinking,
                        "round": round_num,
                    }

                # 每个工具只执行一次, 结果同时用于 UI 展示和 LLM 回传
                round_results = []
                for tc_dict in resp.tool_calls:
                    tc = ToolCall(
                        tool_name=tc_dict["name"],
                        arguments=tc_dict.get("input", tc_dict.get("arguments", {})),
                        call_id=tc_dict.get("id", f"call_{round_num}"),
                    )

                    yield {
                        "type": "tool_call",
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                        "round": round_num,
                    }

                    result = self.registry.execute(tc)
                    all_results.append(result)

                    yield {
                        "type": "tool_result",
                        "tool_name": tc.tool_name,
                        "success": result.success,
                        "summary": result.summary() if result.success else result.error,
                        "round": round_num,
                    }

                    round_results.append({
                        "tool_name": tc.tool_name,
                        "call_id": tc.call_id,
                        "success": result.success,
                        "data": result.data,
                        "error": result.error,
                    })

                messages.append({
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": resp.tool_calls,
                })
                last_results = round_results
                continue

            # LLM 返回了最终答案 → 最终校验 + 低分重答
            elapsed = (datetime.now() - t_start).total_seconds()

            yield {
                "type": "thinking",
                "content": "基于工具结果生成最终回答…",
                "round": round_num,
            }

            events: list[dict] = []
            final_answer, verification = self._verify_and_regenerate(
                query=query,
                answer=resp.content,
                tool_results=[r.to_dict() for r in all_results],
                intent_type=intent.intent_type.value,
                events=events,
            )
            yield from events
            self._add_to_history("assistant", final_answer)

            yield {
                "type": "answer",
                "content": final_answer,
                "verification": {
                    "passed": verification.passed,
                    "overall_score": round(verification.overall_score, 3),
                },
                "elapsed": elapsed,
                "round": round_num,
            }
            return

        # 达到最大轮数 — 强制生成答案 (同样走最终校验)
        yield {
            "type": "thinking",
            "content": "达到最大轮数，基于已有结果生成答案…",
            "round": max_rounds,
        }

        try:
            final_resp = self.llm_client.chat(
                messages + [{"role": "user", "content": "请基于以上所有工具调用结果，直接给出最终回答。不要再调用工具。"}],
                system_prompt=system_prompt,
            )
            answer = final_resp.content if final_resp and final_resp.content else "抱歉，分析过于复杂，请简化您的问题。"
        except Exception:
            answer = "抱歉，分析过于复杂，请简化您的问题。"

        events = []
        final_answer, verification = self._verify_and_regenerate(
            query=query,
            answer=answer,
            tool_results=[r.to_dict() for r in all_results],
            intent_type=intent.intent_type.value,
            events=events,
        )
        yield from events
        self._add_to_history("assistant", final_answer)

        elapsed = (datetime.now() - t_start).total_seconds()
        yield {
            "type": "answer",
            "content": final_answer,
            "verification": {
                "passed": verification.passed,
                "overall_score": round(verification.overall_score, 3),
            },
            "elapsed": elapsed,
            "round": max_rounds + 1,
        }

    # ================================================================
    # LLM 主导模式 (LLM 决定工具调用)
    # ================================================================

    def _run_with_llm(self, query: str, intent: Intent) -> "AgentResponse":
        """LLM 主导执行: LLM 自主决定调用哪些工具.

        流程:
            1. 发送 query + tools → LLM
            2. LLM 返回 tool_calls → 执行 → 结果回传
            3. 重复直到 LLM 返回最终文本答案
            4. 最终校验 → 低分触发重新生成 → 仍不合格则降级返回
        """
        t_start = datetime.now()
        tools = self.registry.get_tool_schemas()
        system_prompt = self._get_system_prompt()

        messages = [{"role": "user", "content": query}]
        all_results = []
        max_rounds = self.config.max_tool_rounds
        chain_of_thought: list[dict] = []  # 思维链记录

        for round_num in range(1, max_rounds + 1):
            # 调用 LLM
            if round_num == 1:
                resp = self.llm_client.chat_with_tools(
                    messages, tools=tools, system_prompt=system_prompt,
                )
            else:
                resp = self.llm_client.chat_with_tool_results(
                    messages, tool_results=last_results, tools=tools,
                    system_prompt=system_prompt,
                )

            # LLM 返回了 tool_calls → 执行
            if resp.has_tool_calls:
                tool_calls = [
                    ToolCall(
                        tool_name=tc["name"],
                        arguments=tc.get("input", tc.get("arguments", {})),
                        call_id=tc.get("id", f"call_{round_num}"),
                    )
                    for tc in resp.tool_calls
                ]

                # 记录思考过程
                thinking = resp.content.strip() if resp.content else ""
                step = {
                    "round": round_num,
                    "type": "tool_call",
                    "thinking": thinking,
                    "tools": [],
                }

                # 执行工具
                round_results = []
                for tc in tool_calls:
                    result = self.registry.execute(tc)
                    all_results.append(result)
                    round_results.append({
                        "tool_name": tc.tool_name,
                        "call_id": tc.call_id,
                        "success": result.success,
                        "data": result.data,
                        "error": result.error,
                    })
                    step["tools"].append({
                        "name": tc.tool_name,
                        "arguments": tc.arguments,
                        "success": result.success,
                        "summary": result.summary() if result.success else result.error,
                    })

                chain_of_thought.append(step)

                # 保存 LLM 的 tool_calls 消息
                messages.append({
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": resp.tool_calls,
                })
                last_results = round_results

                logger.info(
                    "Round %d: %d tool calls → %d results",
                    round_num, len(tool_calls), len(round_results),
                )
                continue

            # LLM 返回了最终文本答案
            answer = resp.content
            elapsed = (datetime.now() - t_start).total_seconds()

            # 记录最终回答
            chain_of_thought.append({
                "round": round_num,
                "type": "final_answer",
                "thinking": "基于以上工具调用结果，生成最终回答",
            })

            # 最终校验 + 低分重答
            events: list[dict] = []
            final_answer, verification = self._verify_and_regenerate(
                query=query,
                answer=answer,
                tool_results=[r.to_dict() for r in all_results],
                intent_type=intent.intent_type.value,
                events=events,
            )
            chain_of_thought.extend(events)

            self._add_to_history("assistant", final_answer)

            return AgentResponse(
                query=query,
                answer=final_answer,
                intent=intent,
                execution=ExecutionResult(
                    success=True, results=all_results,
                ),
                verification=verification,
                elapsed_seconds=elapsed,
                chain_of_thought=chain_of_thought,
            )

        # 超过最大轮数——强制 LLM 基于已有工具结果生成最终答案
        try:
            final_resp = self.llm_client.chat(
                messages + [{"role": "user", "content": "请基于以上所有工具调用结果，直接给出最终回答。不要再调用工具。"}],
                system_prompt=system_prompt,
            )
            answer = final_resp.content if final_resp and final_resp.content else "抱歉，分析过于复杂，请简化您的问题。"
        except Exception:
            answer = "抱歉，分析过于复杂，请简化您的问题。"

        chain_of_thought.append({
            "round": max_rounds + 1,
            "type": "timeout",
            "thinking": "达到最大工具轮数，基于已有结果强制生成答案",
        })

        events = []
        final_answer, verification = self._verify_and_regenerate(
            query=query,
            answer=answer,
            tool_results=[r.to_dict() for r in all_results],
            intent_type=intent.intent_type.value,
            events=events,
        )
        chain_of_thought.extend(events)
        self._add_to_history("assistant", final_answer)

        elapsed = (datetime.now() - t_start).total_seconds()
        return AgentResponse(
            query=query,
            answer=final_answer,
            intent=intent,
            execution=ExecutionResult(results=all_results),
            verification=verification,
            elapsed_seconds=elapsed,
            chain_of_thought=chain_of_thought,
        )

    # ================================================================
    # 带校验的执行
    # ================================================================

    def _execute_with_verification(
        self,
        plan: Plan,
        query: str,
        intent: Intent,
    ) -> ExecutionResult:
        """执行 + 中间校验 + 补充查询循环."""
        max_rounds = self.config.max_tool_rounds
        all_results: list = []
        current_plan = plan

        for round_num in range(1, max_rounds + 1):
            # 执行当前计划
            exec_result = self.executor.execute(current_plan)
            all_results.extend(exec_result.results)

            if not self.config.enable_verification:
                break

            # 中间校验
            verification = self.verifier.verify_after_tool_call(
                query=query,
                tool_results=[r.to_dict() for r in all_results],
                intent_type=intent.intent_type.value,
            )

            # 是否需要补充查询?
            if not verification.needs_more_info:
                break

            # 生成补充计划
            if round_num < max_rounds:
                additional_steps = self._suggestions_to_steps(
                    verification.suggested_tool_calls,
                )
                if additional_steps:
                    current_plan = Plan(
                        intent=intent,
                        steps=additional_steps,
                        reasoning=f"补充查询 (round {round_num + 1})",
                    )
                    logger.info(
                        "Round %d: adding %d more tool calls",
                        round_num + 1, len(additional_steps),
                    )
                else:
                    break
            else:
                logger.warning(
                    "Max tool rounds (%d) reached, stopping", max_rounds,
                )
                break

        # 合并所有结果
        return ExecutionResult(
            success=all(r.success for r in all_results),
            results=all_results,
            errors=[r.error for r in all_results if r.error],
        )

    def _suggestions_to_steps(self, suggestions: list[dict]) -> list:
        """将校验建议转换为 PlanStep."""
        from agent_layer.core.planner import PlanStep

        steps = []
        for sug in suggestions:
            if isinstance(sug, dict) and "tool" in sug:
                steps.append(PlanStep(
                    tool_name=sug["tool"],
                    arguments={"query": sug.get("reason", "")},
                    description=sug.get("reason", ""),
                ))
        return steps

    # ================================================================
    # 答案生成
    # ================================================================

    def _generate_answer_with_llm(
        self,
        query: str,
        exec_result: ExecutionResult,
        intent: Intent,
    ) -> str:
        """使用 LLM 生成答案."""
        # 构建 prompt
        tool_results_text = "\n\n".join(
            r.summary() for r in exec_result.results
        )

        messages = [
            {"role": "user", "content": (
                f"用户查询: {query}\n\n"
                f"意图类型: {intent.intent_type.value}\n\n"
                f"工具调用结果:\n{tool_results_text}\n\n"
                f"请基于以上数据，用中文给出专业、客观的分析回答。"
                f"如果数据不足，请说明需要补充哪些信息。"
            )},
        ]

        if self.llm_client:
            response = self.llm_client.chat(
                messages,
                system_prompt=self._get_system_prompt(),
            )
            return response.content

        return self._generate_answer_rule_based(query, exec_result, intent)

    def _generate_answer_rule_based(
        self,
        query: str,
        exec_result: ExecutionResult,
        intent: Intent,
    ) -> str:
        """基于规则生成答案 (不依赖 LLM)."""
        parts = [f"关于您的问题「{query}」的分析结果如下:\n"]

        success_results = exec_result.get_successful_results()
        failed_results = exec_result.get_failed_results()

        if not success_results:
            parts.append("⚠️ 未能获取到相关数据。")
            if failed_results:
                parts.append("\n错误信息:")
                for r in failed_results:
                    parts.append(f"  - {r.tool_name}: {r.error}")
            return "".join(parts)

        # 汇总工具结果
        for r in success_results:
            data = r.data
            if isinstance(data, dict):
                # 提取 summary
                if "summary" in data:
                    s = data["summary"]
                    parts.append(f"\n📊 {s.get('symbol', r.tool_name)}:")
                    for k, v in s.items():
                        if k != "symbol" and v is not None:
                            parts.append(f"  - {k}: {v}")

                # 知识检索结果
                elif "results" in data:
                    parts.append(f"\n📚 相关知识 ({data.get('count', 0)} 条):")
                    for item in data["results"][:3]:
                        parts.append(
                            f"  - [{item.get('score', 0):.2f}] "
                            f"{item.get('content', '')[:100]}..."
                        )

        # 合规提示
        if intent.intent_type in (
            IntentType.INVESTMENT_ADVICE,
            IntentType.COMPLEX,
        ):
            parts.append(
                "\n\n> ⚠️ 免责声明: 以上分析仅供参考，不构成投资建议。"
                "投资有风险，入市需谨慎。过往业绩不代表未来表现。"
            )

        return "".join(parts)

    # ================================================================
    # 最终校验 + 低分重答
    # ================================================================

    def _verify_and_regenerate(
        self,
        query: str,
        answer: str,
        tool_results: list,
        intent_type: str,
        events: Optional[list] = None,
    ) -> tuple[str, VerificationResult]:
        """最终校验 + 低分重答循环.

        分数低于 min_verification_score 时, 把校验 issues 作为反馈拼进 prompt
        重新生成答案 (最多 max_regeneration_rounds 次, 无改善则提前停止);
        仍不合格则降级处理 (追加合规提示与问题说明, 不重写答案)。

        Args:
            events: 可选列表, 收集过程事件 (供流式场景 yield).

        Returns:
            (最终答案, 校验结果).
        """
        verification = self.verifier.verify_final_answer(
            query=query,
            answer=answer,
            tool_results=tool_results,
            intent_type=intent_type,
        )
        best_answer, best_verification = answer, verification

        for attempt in range(1, self.config.max_regeneration_rounds + 1):
            if best_verification.overall_score >= self.config.min_verification_score:
                break
            if not self.llm_client:
                break
            feedback = self._build_regeneration_prompt(
                query, best_answer, best_verification, tool_results,
            )
            if not feedback:
                break

            if events is not None:
                events.append({
                    "type": "thinking",
                    "content": (
                        f"校验分数 {best_verification.overall_score:.2f} 低于阈值 "
                        f"{self.config.min_verification_score}，"
                        f"正在重新生成回答 (第 {attempt}/{self.config.max_regeneration_rounds} 次)…"
                    ),
                    "round": -1,
                })

            try:
                resp = self.llm_client.chat(
                    [{"role": "user", "content": feedback}],
                    system_prompt=self._get_system_prompt(),
                )
            except Exception as e:
                logger.warning("重新生成回答失败: %s", e)
                if events is not None:
                    events.append({
                        "type": "thinking",
                        "content": f"重新生成失败，使用原回答: {e}",
                        "round": -1,
                    })
                break
            candidate = (resp.content or "").strip() if resp else ""
            if not candidate:
                break

            new_verification = self.verifier.verify_final_answer(
                query=query,
                answer=candidate,
                tool_results=tool_results,
                intent_type=intent_type,
            )
            if events is not None:
                events.append({
                    "type": "thinking",
                    "content": (
                        f"第 {attempt} 次重新生成完成，"
                        f"校验分数 {new_verification.overall_score:.2f}"
                    ),
                    "round": -1,
                })

            if new_verification.overall_score > best_verification.overall_score:
                best_answer, best_verification = candidate, new_verification
            else:
                # 无改善, 继续重试意义不大
                logger.info(
                    "Regeneration attempt %d did not improve score (%.2f <= %.2f)",
                    attempt, new_verification.overall_score, best_verification.overall_score,
                )
                break

        # 降级处理: 仍低于阈值或未通过 → 追加说明文本
        if (
            best_verification.overall_score < self.config.min_verification_score
            or not best_verification.passed
        ):
            best_answer = self._apply_compliance_fix(best_answer, best_verification)

        return best_answer, best_verification

    def _build_regeneration_prompt(
        self,
        query: str,
        answer: str,
        verification: VerificationResult,
        tool_results: list,
    ) -> str:
        """把校验发现的问题整理成反馈 prompt, 用于重新生成答案."""
        issues: list[str] = []
        for dim in verification.dimensions:
            if dim.score >= 0.8:
                continue
            for item in dim.issues:
                issues.append(f"[{dim.dimension}] {item}")
            for sug in dim.suggestions:
                issues.append(f"[{dim.dimension}] 建议: {sug}")
        if not issues:
            return ""
        return REGENERATION_PROMPT.format(
            query=query,
            answer=answer[:2000],
            issues="\n".join(f"- {i}" for i in issues[:10]),
            tool_results=format_tool_results_text(tool_results),
        )

    # ================================================================
    # 合规修正
    # ================================================================

    def _apply_compliance_fix(
        self, answer: str, verification: VerificationResult
    ) -> str:
        """根据校验结果对答案做降级修正 (追加提示文本, 不重写答案)."""
        for dim in verification.dimensions:
            if dim.dimension == "compliance" and dim.verdict.value in ("weak", "fail"):
                # 追加风险提示
                if "风险" not in answer and "不构成" not in answer:
                    answer += (
                        "\n\n> ⚠️ 免责声明: 以上分析仅供参考，不构成投资建议。"
                        "投资有风险，入市需谨慎。"
                    )
                    logger.info("Added compliance disclaimer to answer")

            if dim.dimension == "completeness" and dim.verdict.value in ("weak", "needs_more"):
                if dim.issues:
                    answer += f"\n\n> ℹ️ 信息完整性提示: {'; '.join(dim.issues)}"

            if dim.dimension in ("accuracy", "logic") and dim.verdict.value in ("weak", "fail"):
                if dim.issues:
                    answer += (
                        f"\n\n> ℹ️ 质量校验提示 ({dim.dimension}): "
                        f"{'; '.join(dim.issues[:3])}"
                    )

        return answer

    # ================================================================
    # 辅助
    # ================================================================

    def _get_system_prompt(self) -> str:
        return self.config.system_prompt or FINANCIAL_ADVISOR_PROMPT

    def _add_to_history(self, role: str, content: str) -> None:
        self._conversation_history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        self._conversation_history.clear()

    @property
    def history(self) -> list[dict]:
        return list(self._conversation_history)


# ================================================================
# AgentResponse
# ================================================================

from dataclasses import dataclass, field


@dataclass
class AgentResponse:
    """Agent 完整响应.

    Attributes:
        query: 原始查询.
        answer: 最终答案.
        intent: 识别的意图.
        plan: 工具调用计划.
        execution: 执行结果.
        verification: 校验结果.
        elapsed_seconds: 总耗时 (秒).
        chain_of_thought: 思维链 [{"round": int, "type": str, "thinking": str, "tools": [...]}].
    """

    query: str
    answer: str = ""
    intent: Optional[Intent] = None
    plan: Optional[Plan] = None
    execution: Optional[ExecutionResult] = None
    verification: Optional[VerificationResult] = None
    elapsed_seconds: float = 0.0
    chain_of_thought: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "intent": self.intent.intent_type.value if self.intent else "",
            "tool_calls": len(self.execution.results) if self.execution else 0,
            "verification_passed": self.verification.passed if self.verification else False,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def __repr__(self) -> str:
        return (
            f"AgentResponse(query='{self.query[:50]}...', "
            f"intent={self.intent.intent_type.value if self.intent else '?'}, "
            f"tools={len(self.execution.results) if self.execution else 0}, "
            f"time={self.elapsed_seconds:.1f}s)"
        )
