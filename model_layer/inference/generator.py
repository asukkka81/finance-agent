# model_layer/inference/generator.py
"""模型推理生成器 — vLLM 引擎 + HuggingFace 回退.

支持两种推理模式:
    1. vLLM 高性能模式 (生产环境)
       - Prefix Caching (共享 system prompt)
       - 连续批处理 (Continuous Batching)
       - INT8/FP8 量化加速

    2. HuggingFace 标准模式 (开发/测试)
       - 简单直接，无需额外服务
       - 适合单条推理

Usage::

    gen = ModelGenerator(config, model_path="outputs/merged_model")
    response = gen.generate(
        "请分析贵州茅台的投资价值",
        tools=registry.get_tool_schemas(),
    )
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelGenerator:
    """模型推理生成器.

    封装 vLLM / HuggingFace 推理引擎，
    提供统一的 generate() 接口。
    """

    def __init__(
        self,
        config,
        model_path: Optional[str] = None,
        use_vllm: bool = False,
    ):
        self.config = config
        self.model_path = model_path or config.merged_model_dir
        self.use_vllm = use_vllm
        self._engine = None
        self._tokenizer = None

    # ================================================================
    # 初始化
    # ================================================================

    def initialize(self):
        """初始化推理引擎."""
        if self.use_vllm:
            self._init_vllm()
        else:
            self._init_huggingface()

    def _init_vllm(self):
        """初始化 vLLM 推理引擎."""
        try:
            from vllm import LLM, SamplingParams

            logger.info("Initializing vLLM engine: %s", self.model_path)

            self._engine = LLM(
                model=self.model_path,
                tensor_parallel_size=self.config.vllm_tensor_parallel,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=self.config.vllm_max_model_len,
                trust_remote_code=self.config.trust_remote_code,
                enable_prefix_caching=True,  # 共享 system prompt
                dtype="bfloat16",
            )

            self._sampling_params = SamplingParams(
                temperature=self.config.inference_temperature,
                top_p=self.config.inference_top_p,
                max_tokens=self.config.inference_max_tokens,
            )

            logger.info("vLLM engine initialized")

        except ImportError:
            logger.warning(
                "vLLM not installed. Falling back to HuggingFace. "
                "Install with: pip install vllm"
            )
            self.use_vllm = False
            self._init_huggingface()
        except Exception as e:
            logger.error("vLLM initialization failed: %s", e)
            self.use_vllm = False
            self._init_huggingface()

    def _init_huggingface(self):
        """初始化 HuggingFace 推理."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info("Loading model with HuggingFace: %s", self.model_path)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.config.trust_remote_code,
            )

            self._engine = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=self.config.trust_remote_code,
            )

            logger.info("HuggingFace model loaded")

        except Exception as e:
            logger.error("HuggingFace model loading failed: %s", e)
            raise

    # ================================================================
    # 生成
    # ================================================================

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> dict:
        """生成回答.

        Args:
            prompt: 用户输入.
            system_prompt: 系统提示词.
            tools: 工具定义列表 (Anthropic format).

        Returns:
            {"text": "生成的回答", "tool_calls": [...], "usage": {...}}
        """
        if self._engine is None:
            self.initialize()

        full_prompt = self._format_prompt(prompt, system_prompt, tools)

        if self.use_vllm:
            return self._generate_vllm(full_prompt)
        else:
            return self._generate_hf(full_prompt)

    def generate_batch(
        self, prompts: list[str], **kwargs
    ) -> list[dict]:
        """批量生成."""
        if self._engine is None:
            self.initialize()

        full_prompts = [self._format_prompt(p, **kwargs) for p in prompts]

        if self.use_vllm:
            results = []
            for fp in full_prompts:
                results.append(self._generate_vllm(fp))
            return results
        else:
            return [self._generate_hf(fp) for fp in full_prompts]

    # ================================================================
    # 内部
    # ================================================================

    def _format_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict]] = None,
    ) -> str:
        """格式化输入为 Qwen Chat Template."""
        sp = system_prompt or (
            "你是专业的金融智能顾问。你可以调用工具获取实时数据、"
            "搜索知识库、执行 Python 代码。回答时请引用数据来源，"
            "涉及投资建议时必须包含风险提示。"
        )

        parts = [f"<|im_start|>system\n{sp}<|im_end|>"]

        if tools:
            tool_desc = "可用工具:\n" + "\n".join(
                f"- {t.get('name', '')}: {t.get('description', '')[:100]}"
                for t in tools
            )
            parts.append(f"<|im_start|>system\n{tool_desc}<|im_end|>")

        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def _generate_vllm(self, full_prompt: str) -> dict:
        """vLLM 生成."""
        from vllm import SamplingParams

        outputs = self._engine.generate(
            [full_prompt],
            self._sampling_params,
        )

        text = outputs[0].outputs[0].text

        return {
            "text": text,
            "tool_calls": self._extract_tool_calls(text),
            "usage": {},
        }

    def _generate_hf(self, full_prompt: str) -> dict:
        """HuggingFace 生成."""
        import torch

        inputs = self._tokenizer(
            full_prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self._engine.device)

        with torch.no_grad():
            outputs = self._engine.generate(
                **inputs,
                max_new_tokens=self.config.inference_max_tokens,
                temperature=self.config.inference_temperature,
                top_p=self.config.inference_top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(response_ids, skip_special_tokens=True)

        return {
            "text": text,
            "tool_calls": self._extract_tool_calls(text),
            "usage": {},
        }

    @staticmethod
    def _extract_tool_calls(text: str) -> list[dict]:
        """从生成文本中提取工具调用."""
        import re
        import json

        calls = []
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                items = parsed if isinstance(parsed, list) else [parsed]
                for item in items:
                    if isinstance(item, dict) and "name" in item:
                        calls.append(item)
            except json.JSONDecodeError:
                pass
        return calls

    def unload(self):
        """释放模型资源."""
        if self._engine is not None:
            del self._engine
            self._engine = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        import gc
        gc.collect()
        logger.info("Model unloaded")
