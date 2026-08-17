# agent_layer/config.py
"""Agent 层配置."""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Agent 全局配置.

    Attributes:
        model: LLM 模型名称.
        max_tool_rounds: 最大工具调用轮数.
        enable_verification: 是否启用多轮校验.
        enable_parallel_tools: 是否启用并行工具调用.
        temperature: LLM 温度参数.
        max_tokens: LLM 最大输出 token.
    """

    # --- LLM ---
    model: str = "claude-sonnet-5-20251001"
    temperature: float = 0.1
    max_tokens: int = 4096

    # --- Agent 行为 ---
    max_tool_rounds: int = 5         # 防止无限循环
    enable_verification: bool = True  # 多轮校验
    enable_parallel_tools: bool = True

    # --- 意图识别 ---
    intent_confidence_threshold: float = 0.6

    # --- 校验 ---
    verification_dimensions: list[str] = field(default_factory=lambda: [
        "completeness",   # 信息完整性
        "accuracy",       # 数据准确性
        "logic",          # 逻辑一致性
        "compliance",     # 合规安全性
    ])

    # --- 系统提示词 ---
    system_prompt: str = "financial_advisor"  # 预设角色
