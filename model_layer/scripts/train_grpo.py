#!/usr/bin/env python3
# model_layer/scripts/train_grpo.py
"""GRPO 对齐训练入口.

Usage:
    python -m model_layer.scripts.train_grpo \
        --sft_checkpoint outputs/sft_checkpoint \
        --prompts data/training/grpo_prompts.jsonl \
        --epochs 2 \
        --rollouts 4
"""

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GRPO 强化学习对齐")
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--sft_checkpoint", type=str,
                        default="outputs/sft_checkpoint",
                        help="SFT LoRA checkpoint 路径")
    parser.add_argument("--prompts", type=str,
                        default="data/training/grpo_prompts.jsonl",
                        help="GRPO 训练 prompt 文件")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=4,
                        help="每个 prompt 生成的候选数 (K)")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--output_dir", type=str,
                        default="outputs/grpo_checkpoint")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from model_layer.config import ModelConfig
    from model_layer.grpo.grpo_trainer import GRPOTrainer

    config = ModelConfig(
        base_model=args.base_model,
        num_epochs_grpo=args.epochs,
        num_rollouts_per_prompt=args.rollouts,
        grpo_learning_rate=args.learning_rate,
        grpo_checkpoint_dir=args.output_dir,
    )

    # 加载 prompts
    prompts = []
    if os.path.exists(args.prompts):
        with open(args.prompts, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                prompts.append(record.get("prompt", ""))
        logger.info("Loaded %d GRPO prompts", len(prompts))
    else:
        logger.warning("Prompts file not found: %s. Using defaults.", args.prompts)
        prompts = [
            "请分析贵州茅台的投资价值",
            "什么是夏普比率？如何用它来评估基金？",
            "帮我构建一个保守型投资组合",
            "最新的宏观经济数据说明了什么趋势？",
            "计算一个3只股票组合的有效前沿",
        ]

    # 训练
    trainer = GRPOTrainer(config)

    if args.dry_run:
        logger.info("=== GRPO DRY RUN ===")
        logger.info("Base model: %s", args.base_model)
        logger.info("SFT checkpoint: %s", args.sft_checkpoint)
        logger.info("Prompts: %d, Rollouts: %d, Epochs: %d",
                    len(prompts), args.rollouts, args.epochs)
        logger.info("Reward weights: %s", config.reward_weights)
        logger.info("Output: %s", args.output_dir)
        logger.info("=== DRY RUN COMPLETE ===")
        return

    trainer.initialize(args.sft_checkpoint)
    result = trainer.train(prompts)
    logger.info("GRPO result: %s", result)


if __name__ == "__main__":
    main()
