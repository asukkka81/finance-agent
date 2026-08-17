#!/usr/bin/env python3
# model_layer/scripts/prepare_data.py
"""训练数据准备脚本 — 合成 + 格式化 + 分割.

Usage:
    python -m model_layer.scripts.prepare_data \
        --total 3000 \
        --output_dir data/training/
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="准备训练数据")
    parser.add_argument("--total", type=int, default=3000,
                        help="总合成样本数")
    parser.add_argument("--output_dir", type=str, default="data/training/",
                        help="输出目录")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="测试集比例")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from model_layer.data.synthesizer import DataSynthesizer
    from model_layer.data.dataset import FinanceDataset

    # 合成数据
    counts = {
        "stock_price": args.total * 35 // 100,
        "knowledge_qa": args.total * 35 // 100,
        "tool_chain": args.total * 15 // 100,
        "sandbox": args.total * 15 // 100,
    }

    logger.info("Generating %d synthetic examples...", args.total)

    synth = DataSynthesizer(seed=42)
    dataset = FinanceDataset()
    dataset.load_from_synthesizer(synth, counts)

    logger.info("Dataset stats: %s", dataset.stats)

    # 分割
    train, test = dataset.train_test_split(test_ratio=args.test_ratio)

    # 保存
    train_path = os.path.join(args.output_dir, "sft_train.jsonl")
    eval_path = os.path.join(args.output_dir, "sft_eval.jsonl")

    train.to_jsonl(train_path)
    test.to_jsonl(eval_path)

    logger.info("Train: %d examples → %s", len(train), train_path)
    logger.info("Eval:  %d examples → %s", len(test), eval_path)

    # 保存 GRPO prompts
    grpo_path = os.path.join(args.output_dir, "grpo_prompts.jsonl")
    import json
    with open(grpo_path, "w", encoding="utf-8") as f:
        for ex in test.examples[:100]:  # GRPO 用 100 个 prompt
            user_msgs = [m["content"] for m in ex.messages if m["role"] == "user"]
            if user_msgs:
                f.write(json.dumps({"prompt": user_msgs[0]}, ensure_ascii=False) + "\n")

    logger.info("GRPO prompts: %d → %s", min(100, len(test)), grpo_path)
    logger.info("Done!")


if __name__ == "__main__":
    main()
