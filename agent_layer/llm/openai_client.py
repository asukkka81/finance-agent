# agent_layer/llm/openai_client.py
"""OpenAI 兼容 API 客户端 — 支持阿里云 MAAS / 任意兼容端点.

支持:
    - 基本对话 (chat)
    - 带工具调用的对话 (chat_with_tools)
    - Tool result 回传
    - 任意 OpenAI 兼容端点 (阿里云 MAAS / vLLM / Ollama / ...)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from agent_layer.llm.base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI 兼容 API 客户端.

    支持阿里云模型服务、vLLM、Ollama 等所有 OpenAI 兼容端点。

    Usage::

        client = OpenAIClient(
            api_key="sk-xxx",
            base_url="https://your-endpoint/compatible-mode/v1",
            model="qwen-plus",
        )
        resp = client.chat([{"role": "user", "content": "你好"}])
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "qwen-plus",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        """延迟初始化 OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                logger.info(
                    "OpenAI client initialized: %s (model=%s)",
                    self.base_url, self.model,
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
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
        enriched = list(messages)

        for result in tool_results:
            tool_name = result.get("tool_name", result.get("name", "unknown"))
            call_id = result.get("call_id", "")
            data = result.get("data", "")

            # 序列化数据
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, ensure_ascii=False, default=str)
            else:
                data_str = str(data)

            # 截断过长数据
            if len(data_str) > 4000:
                data_str = data_str[:4000] + f"... (truncated, total {len(data_str)} chars)"

            enriched.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": data_str,
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
        """调用 OpenAI 兼容 API."""
        # 转换消息格式
        api_messages = []

        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # 处理 tool_calls (assistant 消息)
            if role == "assistant" and msg.get("tool_calls"):
                tool_calls_formatted = []
                for tc in msg["tool_calls"]:
                    tool_calls_formatted.append({
                        "id": tc.get("id", tc.get("call_id", "")),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", tc.get("tool_name", "")),
                            "arguments": json.dumps(
                                tc.get("arguments", tc.get("input", {})),
                                ensure_ascii=False,
                            ),
                        },
                    })
                api_messages.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": tool_calls_formatted,
                })
                continue

            # 处理 tool 角色 (工具结果)
            if role == "tool":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("call_id", msg.get("tool_call_id", "")),
                    "content": str(content),
                })
                continue

            # 普通消息
            if isinstance(content, str):
                api_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Anthropic content block 格式 → 提取文本
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                api_messages.append({"role": role, "content": "\n".join(text_parts)})
            else:
                api_messages.append({"role": role, "content": str(content)})

        # 转换工具格式 (Anthropic → OpenAI)
        openai_tools = None
        if tools:
            openai_tools = []
            for t in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", t.get("parameters", {})),
                    },
                })

        params = {
            "model": kwargs.get("model", self.model),
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if openai_tools:
            params["tools"] = openai_tools
            params["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**params)
            return self._parse_response(response)
        except Exception as e:
            logger.error("API call failed: %s", e)
            return LLMResponse(
                content=f"API 调用失败: {e}",
                finish_reason="error",
            )

    def _parse_response(self, response) -> LLMResponse:
        """解析 OpenAI API 响应."""
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or "stop"

        tool_calls = []
        text_content = message.content or ""

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": arguments,
                    "arguments": arguments,
                })

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason="tool_use" if tool_calls else finish_reason,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )
