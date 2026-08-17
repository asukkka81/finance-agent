# model_layer/grpo/__init__.py
"""GRPO 强化学习对齐模块."""

from model_layer.grpo.reward import RewardCalculator
from model_layer.grpo.grpo_trainer import GRPOTrainer

__all__ = ["RewardCalculator", "GRPOTrainer"]
