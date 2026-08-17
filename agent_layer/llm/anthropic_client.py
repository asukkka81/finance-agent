# agent_layer/llm/anthropic_client.py
"""Anthropic Claude API 客户端."""

import json
import logging
from typing import Optional

from agent_layer.llm.base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API 客户端.

    使用 Messages API + tool_use 功能。
    需要 ANTHROPIC_API_KEY 环境变量。

    Usage::

        client = AnthropicClient(model="claude-sonnet-5-20251001")
        resp = client.chat_with_tools(messages, tools, system_prompt)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5-20251001",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        """延迟初始化 Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                import os
                key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
                if not key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY not found. "
                        "Set the environment variable or pass api_key parameter."
                    )
                self._client = anthropic.Anthropic(api_key=key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._client

    # ================================================================
    # 对话接口
    # ================================================================

    def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """基本对话 (无工具)."""
        return self._call_api(messages, system_prompt, tools=None, **kwargs)

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """带工具调用的对话."""
        return self._call_api(messages, system_prompt, tools=tools, **kwargs)

    def chat_with_tool_results(
        self,
        messages: list[dict],
        tool_results: list[dict],
        tools: list[dict],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """将工具结果回传给 LLM."""
        # 在 messages 末尾追加 tool_result blocks
        enriched = list(messages)

        for result in tool_results:
            enriched.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.get("call_id", ""),
                        "content": json.dumps(
                            result.get("data", ""),
                            ensure_ascii=False,
                            default=str,
                        ),
                        "is_error": not result.get("success", True),
                    }
                ],
            })

        return self._call_api(enriched, system_prompt, tools=tools, **kwargs)

    # ================================================================
    # 内部
    # ================================================================

    def _call_api(
        self,
        messages: list[dict],
        system_prompt: Optional[str],
        tools: Optional[list[dict]],
        **kwargs,
    ) -> LLMResponse:
        """调用 Anthropic Messages API."""
        # 转换消息格式
        anthropic_messages = self._convert_messages(messages)

        params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": anthropic_messages,
        }

        if system_prompt:
            params["system"] = system_prompt

        if tools:
            params["tools"] = tools

        try:
            response = self.client.messages.create(**params)
            return self._parse_response(response)
        except Exception as e:
            logger.error("Anthropic API call failed: %s", e)
            return LLMResponse(
                content=f"API 调用失败: {e}",
                finish_reason="error",
            )

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """转换内部消息格式为 Anthropic Messages 格式."""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                converted.append({"role": role, "content": content})
            elif isinstance(content, list):
                # 已经是 Anthropic content block 格式
                converted.append({"role": role, "content": content})
            else:
                converted.append({"role": role, "content": str(content)})

        return converted

    def _parse_response(self, response) -> LLMResponse:
        """解析 Anthropic API 响应."""
        tool_calls = []
        text_content = []

        for block in response.content:
            if block.type == "text":
                text_content.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return LLMResponse(
            content="\n".join(text_content),
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            usage={
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            },
        )
