#!/usr/bin/env python3
# model_layer/scripts/train_sft.py
"""LoRA SFT 训练入口.

Usage:
    python -m model_layer.scripts.train_sft \
        --base_model Qwen/Qwen2.5-14B-Instruct \
        --train_data data/training/sft_train.jsonl \
        --eval_data data/training/sft_eval.jsonl \
        --epochs 3 \
        --output_dir outputs/sft_checkpoint
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT 训练")
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--train_data", type=str,
                        default="data/training/sft_train.jsonl")
    parser.add_argument("--eval_data", type=str,
                        default="data/training/sft_eval.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str,
                        default="outputs/sft_checkpoint")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅验证流程，不实际训练")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from model_layer.config import ModelConfig
    from model_layer.data.dataset import FinanceDataset
    from model_layer.sft.trainer import SFTTrainer
    from model_layer.sft.lora_config import get_lora_config, estimate_trainable_params

    # 配置
    config = ModelConfig(
        base_model=args.base_model,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        num_epochs_sft=args.epochs,
        sft_checkpoint_dir=args.output_dir,
    )

    # LoRA 参数估计
    lora_cfg = get_lora_config(config)
    params = estimate_trainable_params(args.base_model, lora_cfg)
    logger.info("LoRA Config: r=%d, alpha=%d, target_modules=%d",
                args.lora_r, config.lora_alpha, len(config.lora_target_modules))
    logger.info("参数估计: 总参数=%.1fB, 可训练=%.1fM (%.4f%%)",
                params["total_params"] / 1e9,
                params["trainable_params"] / 1e6,
                params["ratio"])

    # 加载数据
    dataset = FinanceDataset()
    if os.path.exists(args.train_data):
        dataset.load_jsonl(args.train_data)
        logger.info("Loaded %d training examples", len(dataset))
    else:
        logger.warning("训练数据不存在: %s。将使用合成数据。", args.train_data)
        from model_layer.data.synthesizer import DataSynthesizer
        synth = DataSynthesizer()
        dataset.load_from_synthesizer(synth, {"stock_price": 500, "knowledge_qa": 500})

    train_data = dataset.to_huggingface()
    logger.info("Training samples: %d", len(train_data))

    eval_data = None
    if os.path.exists(args.eval_data):
        eval_dataset = FinanceDataset().load_jsonl(args.eval_data)
        eval_data = eval_dataset.to_huggingface()

    # 训练
    trainer = SFTTrainer(config)

    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info("Base model: %s", args.base_model)
        logger.info("Training samples: %d", len(train_data))
        logger.info("Epochs: %d", args.epochs)
        logger.info("LoRA rank: %d, LR: %.2e", args.lora_r, args.learning_rate)
        logger.info("Trainable params: %.2fM", params["trainable_params"] / 1e6)
        logger.info("Estimated GPU memory: ~24GB (bf16 + LoRA + batch)")
        logger.info("=== DRY RUN COMPLETE ===")
        return

    result = trainer.train_simple(train_data, eval_data)
    logger.info("Training result: %s", result)


if __name__ == "__main__":
    main()
