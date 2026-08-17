# agent_layer/core/planner.py
"""策略规划器 — 将意图转换为工具调用计划.

支持三种调度模式:
    1. 串行执行 (SERIAL): 工具A → 工具B → 工具C (有依赖)
    2. 并行调用 (PARALLEL): 工具A ‖ 工具B ‖ 工具C (无依赖)
    3. 条件触发 (CONDITIONAL): 如果A结果满足条件 → 执行B

规划策略:
    1. 基于意图类型的模板匹配 (快速)
    2. LLM 动态规划 (灵活，用于复杂意图)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from agent_layer.core.intent import Intent, IntentType

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    SERIAL = "serial"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class PlanStep:
    """一个工具调用步骤.

    Attributes:
        tool_name: 工具名称 (对应 ToolRegistry).
        arguments: 工具参数.
        depends_on: 依赖的步骤索引 (None = 无依赖).
        condition: 条件函数 (用于条件触发).
        description: 人类可读的步骤描述.
    """

    tool_name: str
    arguments: dict = field(default_factory=dict)
    depends_on: Optional[int] = None  # 依赖的 step index
    condition: Optional[str] = None
    description: str = ""


@dataclass
class Plan:
    """工具调用计划.

    Attributes:
        intent: 原始意图.
        steps: 工具调用步骤列表.
        mode: 执行模式.
        reasoning: 规划理由 (LLM 生成时).
    """

    intent: Intent
    steps: list[PlanStep] = field(default_factory=list)
    mode: ExecutionMode = ExecutionMode.SERIAL
    reasoning: str = ""

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def get_parallel_groups(self) -> list[list[int]]:
        """获取可并行的步骤组.

        返回 [[step_indices], ...] 每组内可并行，组间顺序执行.
        """
        if self.mode != ExecutionMode.PARALLEL:
            return [[i] for i in range(len(self.steps))]

        # 简单策略: 无依赖的步骤放第一组，后续步骤顺序执行
        independent = []
        dependent = []
        for i, step in enumerate(self.steps):
            if step.depends_on is None:
                independent.append(i)
            else:
                dependent.append(i)

        groups = []
        if independent:
            groups.append(independent)
        for idx in dependent:
            groups.append([idx])

        return groups


class Planner:
    """策略规划器.

    职责: 根据用户意图生成工具调用计划。

    两种模式:
        - Template (模板匹配): 已知意图 → 预定义模板
        - LLM (动态规划): 复杂意图 → LLM 推理生成计划
    """

    # ================================================================
    # 意图 → 计划模板
    # ================================================================

    PLAN_TEMPLATES: dict[IntentType, list[dict]] = {
        IntentType.STOCK_PRICE: [
            {
                "tool": "get_stock_price",
                "args_from": "entities.symbol[0], entities.time_range",
                "description": "获取股票行情数据",
            },
        ],
        IntentType.STOCK_SEARCH: [
            {
                "tool": "search_stocks",
                "args_from": "query",
                "description": "搜索匹配的股票",
            },
        ],
        IntentType.KNOWLEDGE_QA: [
            {
                "tool": "search_knowledge",
                "args_from": "query",
                "description": "从知识库检索相关金融知识",
            },
        ],
        IntentType.MACRO_ANALYSIS: [
            {
                "tool": "get_macro_indicator",
                "args_from": "entities.indicator[0]",
                "description": "获取宏观经济指标",
            },
            {
                "tool": "search_knowledge",
                "args_from": "query",
                "description": "获取相关分析知识",
                "depends_on": 0,
            },
        ],
        IntentType.FUND_QUERY: [
            {
                "tool": "search_knowledge",
                "args_from": "query",
                "description": "搜索基金信息",
            },
        ],
        IntentType.INVESTMENT_ADVICE: [
            {
                "tool": "search_knowledge",
                "args_from": "query",
                "description": "检索投资策略和配置方案",
            },
        ],
        IntentType.PORTFOLIO: [
            {
                "tool": "get_stock_price",
                "args_from": "entities.symbol",
                "description": "获取持仓股票行情",
            },
            {
                "tool": "search_knowledge",
                "args_from": "query",
                "description": "获取分析建议",
                "depends_on": 0,
            },
        ],
        IntentType.GENERAL: [
            {
                "tool": "search_knowledge",
                "args_from": "query",
                "description": "通用知识检索",
            },
        ],
        IntentType.COMPLEX: [
            # COMPLEX 意图交给 LLM 动态规划
        ],
    }

    def __init__(self, use_llm: bool = False, llm_client=None):
        self.use_llm = use_llm
        self.llm_client = llm_client

    # ================================================================
    # 规划
    # ================================================================

    def plan(self, intent: Intent) -> Plan:
        """根据意图生成计划.

        Args:
            intent: 解析后的用户意图.

        Returns:
            Plan 对象.
        """
        # 模板匹配
        templates = self.PLAN_TEMPLATES.get(intent.intent_type, [])

        if templates and not (
            self.use_llm and intent.intent_type == IntentType.COMPLEX
        ):
            return self._plan_from_template(intent, templates)

        # LLM 动态规划 (复杂意图)
        if self.use_llm and self.llm_client:
            return self._plan_with_llm(intent)

        # 兜底: 通用知识检索
        return self._plan_from_template(
            intent,
            self.PLAN_TEMPLATES[IntentType.GENERAL],
        )

    def _plan_from_template(
        self, intent: Intent, templates: list[dict]
    ) -> Plan:
        """从模板构建计划."""
        steps = []
        for tpl in templates:
            args = self._resolve_args(intent, tpl.get("args_from", ""), tpl.get("tool", ""))
            steps.append(PlanStep(
                tool_name=tpl["tool"],
                arguments=args,
                depends_on=tpl.get("depends_on"),
                description=tpl["description"],
            ))

        # 判断执行模式
        has_deps = any(s.depends_on is not None for s in steps)
        mode = ExecutionMode.SERIAL if has_deps else ExecutionMode.PARALLEL

        return Plan(
            intent=intent,
            steps=steps,
            mode=mode,
            reasoning=f"Template-based plan for {intent.intent_type.value}",
        )

    def _plan_with_llm(self, intent: Intent) -> Plan:
        """使用 LLM 动态规划 (用于复杂意图)."""
        # 构建 planning prompt
        from agent_layer.prompts.templates import PLANNING_PROMPT

        prompt = PLANNING_PROMPT.format(
            query=intent.original_query,
            intent_type=intent.intent_type.value,
            entities=intent.entities,
        )

        # 这里需要 tool schemas，实际使用时传入
        logger.info("LLM planning for: %s", intent.original_query[:80])

        # 回退到模板规划
        return Plan(
            intent=intent,
            steps=[
                PlanStep(
                    tool_name="search_knowledge",
                    arguments={"query": intent.original_query, "top_k": 5},
                    description="LLM规划回退：通用检索",
                ),
            ],
            mode=ExecutionMode.SERIAL,
            reasoning="LLM planner not fully configured, fallback to template",
        )

    def _resolve_args(self, intent: Intent, args_spec: str, tool_name: str = "") -> dict:
        """从意图中解析参数.

        Args:
            intent: 解析后的意图.
            args_spec: 参数来源描述.
            tool_name: 目标工具名 (用于匹配正确的参数名).

        Returns:
            {arg_name: value} 字典.
        """
        if "query" in args_spec:
            # 不同工具用不同参数名
            if tool_name == "search_stocks":
                return {"keyword": intent.original_query, "limit": 10}
            else:
                return {"query": intent.original_query, "top_k": 5}

        if "symbol" in args_spec:
            symbols = intent.symbols
            if symbols:
                return {"symbol": symbols[0], "start": "2024-01-01"}

        if "indicator" in args_spec:
            indicators = intent.indicator_names
            if indicators:
                return {"indicator_name": indicators[0]}

        # 默认: 通用搜索
        return {"query": intent.original_query, "top_k": 5}
