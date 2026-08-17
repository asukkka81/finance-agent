# model_layer/sft/lora_config.py
"""LoRA 配置工具 — 生成 PEFT LoRA 配置."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_lora_config(config) -> dict:
    """从 ModelConfig 生成 PEFT LoraConfig 参数.

    Args:
        config: ModelConfig 实例.

    Returns:
        PEFT LoraConfig 参数字典.
    """
    return {
        "r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": config.lora_target_modules,
        "modules_to_save": config.modules_to_save,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def estimate_trainable_params(
    base_model_name: str, lora_config: dict
) -> dict:
    """估算 LoRA 可训练参数量.

    Args:
        base_model_name: 基座模型名.
        lora_config: LoRA 配置.

    Returns:
        {'total': N, 'trainable': M, 'ratio': pct}.
    """
    # Qwen3-14B 约 14B 参数
    model_sizes = {
        "Qwen/Qwen2.5-14B-Instruct": 14_000_000_000,
        "Qwen/Qwen2.5-7B-Instruct": 7_000_000_000,
        "Qwen/Qwen2.5-1.5B-Instruct": 1_500_000_000,
    }

    total = model_sizes.get(base_model_name, 14_000_000_000)

    # LoRA 参数估算
    r = lora_config.get("r", 64)
    target_count = len(lora_config.get("target_modules", []))
    hidden_size = 5120  # Qwen2.5-14B 的 hidden_size

    # 每个 target module 的 LoRA 参数: 2 * hidden_size * r
    trainable = target_count * 2 * hidden_size * r

    return {
        "total_params": total,
        "trainable_params": trainable,
        "ratio": trainable / total * 100,
        "target_modules": target_count,
        "lora_rank": r,
    }


def suggest_batch_size(gpu_memory_gb: float, model_size_b: int = 14) -> int:
    """根据 GPU 显存推荐 batch size.

    Args:
        gpu_memory_gb: GPU 显存 (GB).
        model_size_b: 模型大小 (B 参数).

    Returns:
        推荐的 per_device_batch_size.
    """
    # 经验公式: 14B 模型 + LoRA 约需 24GB (bf16)
    base_memory = model_size_b * 2  # bf16 = 2 bytes per param
    lora_overhead = base_memory * 0.15  # LoRA 额外约 15%
    total_model_memory = base_memory + lora_overhead

    available_for_batch = gpu_memory_gb - total_model_memory

    if available_for_batch < 2:
        return 1
    elif available_for_batch < 4:
        return 1
    elif available_for_batch < 8:
        return 2
    elif available_for_batch < 16:
        return 4
    else:
        return 8
