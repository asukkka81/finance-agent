#!/usr/bin/env python3
# scripts/train_lora_sft.py
"""LoRA SFT 微调启动脚本 — Qwen3 + DISC-FinLLM + MCP 工具轨迹.

使用方法:
  本地 dry run (验证流程):
    python scripts/train_lora_sft.py --dry_run

  GPU 训练 (需要 transformers + peft + torch):
    python scripts/train_lora_sft.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --train_data data/training/sft_train.jsonl \
      --eval_data data/training/sft_eval.jsonl \
      --epochs 3 \
      --lora_r 64 \
      --output_dir outputs/sft_checkpoint

训练配置 (推荐):
  - 模型: Qwen2.5-7B-Instruct (7B 可在单张 A100-40G 上 LoRA 微调)
  - Qwen2.5-14B-Instruct (14B 需要 A100-80G 或双卡)
  - LoRA rank: 64, alpha: 128
  - Batch size: 2~4 (per GPU), 梯度累积 4~8 步
  - 学习率: 2e-4 (cosine schedule)
  - 训练轮数: 3
  - 预计时间: 7B 约 1-2 小时, 14B 约 3-4 小时 (A100)
"""

import argparse, os, json, logging, sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT 金融 Agent 微调")

    # 模型
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="基座模型 (推荐 Qwen2.5-7B-Instruct 或 14B-Instruct)")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    # 数据
    parser.add_argument("--train_data", type=str,
                        default="data/training/sft_train.jsonl")
    parser.add_argument("--eval_data", type=str,
                        default="data/training/sft_eval.jsonl")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank (越大越强但越慢, 推荐 16~128)")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha (通常 = 2× r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 训练
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="per_device batch size")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="梯度累积步数 (有效 batch = batch_size × grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="最大序列长度")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # 输出
    parser.add_argument("--output_dir", type=str,
                        default="outputs/sft_checkpoint")
    parser.add_argument("--merged_dir", type=str,
                        default="outputs/merged_model",
                        help="合并 LoRA 后的完整模型输出路径")

    # 其他
    parser.add_argument("--dry_run", action="store_true",
                        help="仅验证流程，不实际训练")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="使用 4-bit 量化 (QLoRA)，减少显存占用")
    parser.add_argument("--hf_mirror", type=str,
                        default="https://hf-mirror.com",
                        help="HuggingFace 镜像")

    args = parser.parse_args()

    # 设置 HF 镜像
    os.environ["HF_ENDPOINT"] = args.hf_mirror

    # ── 打印训练配置 ──
    log.info("=" * 60)
    log.info("  LoRA SFT 微调配置")
    log.info("=" * 60)
    log.info(f"  基座模型:     {args.model}")
    log.info(f"  LoRA:         r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    log.info(f"  训练数据:     {args.train_data}")
    log.info(f"  验证数据:     {args.eval_data}")
    log.info(f"  训练轮数:     {args.epochs}")
    log.info(f"  Batch size:   {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    log.info(f"  学习率:       {args.lr}")
    log.info(f"  最大长度:     {args.max_length}")
    log.info(f"  量化:         {'4-bit QLoRA' if args.use_4bit else 'full bf16'}")
    log.info(f"  输出目录:     {args.output_dir}")
    log.info(f"  HF 镜像:      {args.hf_mirror}")
    log.info("=" * 60)

    # ── 检查数据 ──
    if not os.path.exists(args.train_data):
        log.error(f"训练数据不存在: {args.train_data}")
        log.error("请先运行: python scripts/prepare_sft_data.py")
        sys.exit(1)

    with open(args.train_data) as f:
        train_count = sum(1 for _ in f)
    log.info(f"\n训练集: {train_count} 条")

    eval_count = 0
    if os.path.exists(args.eval_data):
        with open(args.eval_data) as f:
            eval_count = sum(1 for _ in f)
    log.info(f"验证集: {eval_count} 条")

    # ── 估算资源 ──
    log.info("\n--- 资源估算 ---")
    model_size = 7 if "7B" in args.model else 14 if "14B" in args.model else "?"
    gpu_memory_est = {
        7: "~16GB (4-bit) / ~24GB (bf16)",
        14: "~28GB (4-bit) / ~48GB (bf16)",
    }.get(model_size, "未知")
    log.info(f"  模型大小: {model_size}B")
    log.info(f"  预计显存: {gpu_memory_est}")
    log.info(f"  推荐 GPU: {'A100-40G 或以上' if model_size == 7 else 'A100-80G 或 2×A100-40G'}")
    log.info(f"  预计时间: {'~1-2 小时' if model_size == 7 else '~3-4 小时'} (A100, {args.epochs} epochs)")

    if args.dry_run:
        log.info("\n=== DRY RUN 完成 === (未实际训练)")
        log.info("去掉 --dry_run 参数即可开始真实训练。")
        log.info(f"\n训练命令 (复制到 GPU 服务器):")
        log.info(f"  python scripts/train_lora_sft.py \\")
        log.info(f"    --model {args.model} \\")
        log.info(f"    --epochs {args.epochs} --lora_r {args.lora_r} \\")
        log.info(f"    --output_dir {args.output_dir}")
        return

    # ── 开始训练 ──
    log.info("\n开始训练...")

    try:
        from model_layer.config import ModelConfig
        from model_layer.sft.trainer import SFTTrainer
        from model_layer.sft.lora_config import get_lora_config, estimate_trainable_params
        from datasets import Dataset

        # 配置
        config = ModelConfig(
            base_model=args.model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            learning_rate=args.lr,
            per_device_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_epochs_sft=args.epochs,
            max_seq_length=args.max_length,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            sft_checkpoint_dir=args.output_dir,
            merged_model_dir=args.merged_dir,
        )

        # 加载数据
        train_data = []
        with open(args.train_data) as f:
            for line in f:
                ex = json.loads(line.strip())
                train_data.append(ex)

        eval_data = None
        if os.path.exists(args.eval_data):
            eval_data = []
            with open(args.eval_data) as f:
                for line in f:
                    eval_data.append(json.loads(line.strip()))

        # 训练
        trainer = SFTTrainer(config)
        result = trainer.train_simple(train_data, eval_data)

        if result["status"] == "success":
            log.info(f"\n训练完成! 模型保存至: {args.output_dir}")

            # 合并 LoRA
            log.info("合并 LoRA 权重...")
            trainer.merge_and_save(args.merged_dir)
            log.info(f"合并模型保存至: {args.merged_dir}")
        else:
            log.warning(f"训练结果: {result}")

    except ImportError as e:
        log.error(f"缺少依赖: {e}")
        log.error("安装命令: pip install transformers peft accelerate bitsandbytes torch datasets")
        log.error("")
        log.error("如果没有 GPU，可以租用 AutoDL / 阿里云 PAI / 矩池云等平台的 A100。")
        log.error("将 data/training/ 和 scripts/ 上传到 GPU 服务器后执行上述命令即可。")
        sys.exit(1)


if __name__ == "__main__":
    main()
