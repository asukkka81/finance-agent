# agent_layer/core/executor.py
"""工具执行引擎 — 按计划调度工具调用.

功能:
    1. 执行工具调用计划 (串行/并行/条件)
    2. 处理依赖关系 (前一步的输出 → 后一步的输入)
    3. 错误处理 (工具调用失败 → 回退策略)
    4. 结果收集与格式化
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from agent_layer.core.planner import ExecutionMode, Plan, PlanStep
from agent_layer.mcp.registry import ToolRegistry
from agent_layer.mcp.types import ToolCall, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """执行结果.

    Attributes:
        success: 是否全部成功.
        results: 每个步骤的执行结果.
        total_time_ms: 总耗时 (毫秒).
        errors: 错误列表.
    """

    success: bool = True
    results: list[ToolResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def get_successful_results(self) -> list[ToolResult]:
        return [r for r in self.results if r.success]

    def get_failed_results(self) -> list[ToolResult]:
        return [r for r in self.results if not r.success]

    @property
    def summary(self) -> str:
        success_count = len(self.get_successful_results())
        fail_count = len(self.get_failed_results())
        return (
            f"执行完成: {success_count} 成功, {fail_count} 失败 "
            f"({self.total_time_ms:.0f}ms)"
        )


class ToolExecutor:
    """工具执行引擎.

    执行策略:
        - SERIAL: 按步骤顺序执行，前一步输出可作为后一步输入
        - PARALLEL: 无依赖步骤并发执行
        - CONDITIONAL: 检查条件后决定是否执行后续步骤
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    # ================================================================
    # 执行
    # ================================================================

    def execute(self, plan: Plan) -> ExecutionResult:
        """执行工具调用计划.

        Args:
            plan: Planner 生成的 Plan.

        Returns:
            ExecutionResult 包含所有步骤的执行结果.
        """
        import time
        t0 = time.time()

        if plan.mode == ExecutionMode.PARALLEL:
            results = self._execute_parallel(plan)
        else:
            results = self._execute_serial(plan)

        elapsed = (time.time() - t0) * 1000

        errors = [r.error for r in results if not r.success and r.error]
        all_success = len(errors) == 0

        result = ExecutionResult(
            success=all_success,
            results=results,
            total_time_ms=elapsed,
            errors=errors,
        )

        logger.info(result.summary)
        return result

    def _execute_serial(self, plan: Plan) -> list[ToolResult]:
        """串行执行 — 步骤按序执行，结果可在步骤间传递."""
        results: list[ToolResult] = []
        step_outputs: dict[int, ToolResult] = {}

        for i, step in enumerate(plan.steps):
            # 检查依赖
            if step.depends_on is not None:
                dep_result = step_outputs.get(step.depends_on)
                if dep_result is None or not dep_result.success:
                    results.append(ToolResult(
                        tool_name=step.tool_name,
                        success=False,
                        error=f"依赖步骤 {step.depends_on} 未成功执行",
                    ))
                    continue

                # 将依赖步骤的结果注入参数
                step.arguments = self._enrich_args(
                    step.arguments, dep_result.data,
                )

            # 条件检查
            if step.condition:
                if not self._check_condition(step.condition, step_outputs):
                    logger.debug("Condition not met, skipping step: %s", step.description)
                    continue

            # 执行
            call = ToolCall(tool_name=step.tool_name, arguments=step.arguments)
            result = self.registry.execute(call)
            results.append(result)
            step_outputs[i] = result

            # 步骤失败 → 是否继续?
            if not result.success:
                logger.warning(
                    "Step %d '%s' failed: %s",
                    i, step.tool_name, result.error,
                )
                # 非关键步骤失败不中断
                # (可在 Planner 中标记 is_critical)

        return results

    def _execute_parallel(self, plan: Plan) -> list[ToolResult]:
        """并行执行 — 按依赖关系分组，组内并行."""
        groups = plan.get_parallel_groups()

        all_results: list[ToolResult] = []
        results_map: dict[int, ToolResult] = {}

        for group in groups:
            # 组内可以并行 (当前实现为顺序，后续可改为 asyncio.gather)
            for idx in group:
                step = plan.steps[idx]

                # 处理依赖
                if step.depends_on is not None and step.depends_on in results_map:
                    dep_result = results_map[step.depends_on]
                    if dep_result.success:
                        step.arguments = self._enrich_args(
                            step.arguments, dep_result.data,
                        )

                call = ToolCall(tool_name=step.tool_name, arguments=step.arguments)
                result = self.registry.execute(call)
                all_results.append(result)
                results_map[idx] = result

        return all_results

    # ================================================================
    # 辅助
    # ================================================================

    def _enrich_args(self, args: dict, dep_data: any) -> dict:
        """用依赖步骤的结果丰富参数.

        例如: 前一步 search_stocks 返回了 symbol=600519，
              后一步 get_stock_price 自动填入 symbol 参数。
        """
        enriched = dict(args)

        if dep_data is None:
            return enriched

        if isinstance(dep_data, dict):
            # 自动填入缺失参数
            if "symbol" not in enriched and "symbol" in dep_data:
                enriched["symbol"] = dep_data["symbol"]
            if "stock_id" not in enriched and "stock_id" in dep_data:
                enriched["stock_id"] = dep_data["stock_id"]

            # 从 results 列表中提取
            if "results" in dep_data and isinstance(dep_data["results"], list):
                first = dep_data["results"][0] if dep_data["results"] else {}
                if "symbol" not in enriched and "symbol" in first:
                    enriched["symbol"] = first["symbol"]

        return enriched

    def _check_condition(
        self, condition: str, results: dict[int, ToolResult]
    ) -> bool:
        """检查条件表达式 (简化版)."""
        # 简单实现: 条件总是 True
        # 完整实现应解析条件表达式
        return True
