# agent_layer/llm/base.py
"""LLM 客户端抽象基类."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMResponse:
    """LLM 返回的标准化响应.

    Attributes:
        content: 文本内容 (可能是工具调用结果、最终答案等).
        tool_calls: LLM 请求的工具调用列表.
        finish_reason: 结束原因: 'stop' | 'tool_use' | 'length'.
        usage: token 用量统计.
    """

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_final(self) -> bool:
        return self.finish_reason == "stop" and not self.has_tool_calls


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类.

    支持:
        - 基本对话 (chat)
        - 带工具调用的对话 (chat_with_tools)
        - Tool result 回传

    后续可替换为 Qwen 或其他模型。
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """基本对话 (不含工具).

        Args:
            messages: 对话历史 [{"role": "user", "content": "..."}, ...].
            system_prompt: 系统提示词.

        Returns:
            LLMResponse.
        """
        ...

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """带工具调用的对话.

        Args:
            messages: 对话历史.
            tools: 工具 schema 列表 (OpenAI/Anthropic 格式).
            system_prompt: 系统提示词.

        Returns:
            LLMResponse (可能包含 tool_calls).
        """
        ...

    @abstractmethod
    def chat_with_tool_results(
        self,
        messages: list[dict],
        tool_results: list[dict],
        tools: list[dict],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """将工具执行结果回传给 LLM，让其生成最终答案或下一轮工具调用.

        Args:
            messages: 对话历史 (含 tool_use blocks).
            tool_results: 工具执行结果列表.
            tools: 工具 schema 列表.
            system_prompt: 系统提示词.

        Returns:
            LLMResponse.
        """
        ...
