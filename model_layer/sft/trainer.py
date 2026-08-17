# model_layer/sft/trainer.py
"""LoRA SFT 训练器 — HuggingFace Trainer 封装.

训练流程:
    1. 加载基座模型 (Qwen2.5-14B-Instruct)
    2. 应用 LoRA adapters
    3. Tokenize 数据集
    4. SFT 训练 (Causal LM loss)
    5. 保存 LoRA 权重
    6. (可选) 合并模型

Usage::

    config = ModelConfig()
    trainer = SFTTrainer(config)
    trainer.train(train_dataset, eval_dataset)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class SFTTrainer:
    """LoRA SFT 训练器.

    封装 HuggingFace PEFT + transformers Trainer，
    提供针对金融对话数据优化的训练流程。

    使用条件:
        pip install transformers peft accelerate bitsandbytes datasets
    """

    def __init__(self, config):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._trainer = None

    # ================================================================
    # 模型加载
    # ================================================================

    def load_model(self):
        """加载基座模型 + tokenizer，应用 LoRA.

        Returns:
            (model, tokenizer) 元组.
        """
        from model_layer.sft.lora_config import get_lora_config

        logger.info("Loading base model: %s", self.config.base_model)

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            from peft import LoraConfig, get_peft_model, TaskType

            # 量化配置 (可选 — 减少显存占用)
            bnb_config = None
            if self.config.torch_dtype == "bfloat16":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )

            # Tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="right",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # 模型
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "torch_dtype": torch.bfloat16
                if self.config.torch_dtype == "bfloat16"
                else torch.float16,
            }

            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config

            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                **model_kwargs,
            )

            # 应用 LoRA
            lora_config_dict = get_lora_config(self.config)
            lora_config = LoraConfig(
                r=lora_config_dict["r"],
                lora_alpha=lora_config_dict["lora_alpha"],
                lora_dropout=lora_config_dict["lora_dropout"],
                target_modules=lora_config_dict["target_modules"],
                bias=lora_config_dict["bias"],
                task_type=TaskType.CAUSAL_LM,
            )

            self._model = get_peft_model(base_model, lora_config)
            self._model.print_trainable_parameters()

            logger.info("Model loaded with LoRA adapters")
            return self._model, self._tokenizer

        except ImportError as e:
            logger.error("Required package missing: %s", e)
            logger.error(
                "Install with: pip install transformers peft accelerate "
                "bitsandbytes torch datasets"
            )
            raise
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    # ================================================================
    # 训练
    # ================================================================

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        resume_from_checkpoint: bool = False,
    ):
        """执行 LoRA SFT 训练.

        Args:
            train_dataset: HuggingFace Dataset (含 'text' 列).
            eval_dataset: 验证集 (可选).
            resume_from_checkpoint: 是否从 checkpoint 恢复.
        """
        if self._model is None or self._tokenizer is None:
            self.load_model()

        import torch
        from transformers import (
            TrainingArguments,
            DataCollatorForLanguageModeling,
            Trainer,
        )

        # Tokenize 数据集
        def tokenize_fn(examples):
            texts = examples["text"]
            return self._tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
            )

        train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        if eval_dataset:
            eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.sft_checkpoint_dir,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            num_train_epochs=self.config.num_epochs_sft,
            max_steps=self.config.max_steps_sft if self.config.max_steps_sft > 0 else -1,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            bf16=self.config.torch_dtype == "bfloat16",
            gradient_checkpointing=True,
            report_to="tensorboard",
            run_name="finance-agent-sft",
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        logger.info("Starting SFT training...")
        self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("SFT training complete")

        # 保存 LoRA 权重
        self.save_lora()

    # ================================================================
    # 保存 & 加载
    # ================================================================

    def save_lora(self, path: Optional[str] = None):
        """保存 LoRA adapter 权重."""
        if path is None:
            path = self.config.sft_checkpoint_dir

        if self._model:
            self._model.save_pretrained(path)
            if self._tokenizer:
                self._tokenizer.save_pretrained(path)
            logger.info("LoRA weights saved to %s", path)
        else:
            logger.warning("No model to save")

    def load_lora(self, path: Optional[str] = None):
        """加载已训练的 LoRA 权重."""
        if path is None:
            path = self.config.sft_checkpoint_dir

        from peft import PeftModel
        import torch
        from transformers import AutoModelForCausalLM

        logger.info("Loading LoRA adapter from %s", path)

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=self.config.trust_remote_code,
        )

        self._model = PeftModel.from_pretrained(base_model, path)
        logger.info("LoRA adapter loaded")

    def merge_and_save(self, output_path: Optional[str] = None):
        """合并 LoRA 权重到基座模型并保存."""
        if output_path is None:
            output_path = self.config.merged_model_dir

        if self._model is None:
            self.load_lora()

        logger.info("Merging LoRA weights...")
        merged = self._model.merge_and_unload()

        merged.save_pretrained(output_path, safe_serialization=True)
        if self._tokenizer:
            self._tokenizer.save_pretrained(output_path)

        logger.info("Merged model saved to %s", output_path)

    # ================================================================
    # 简化训练接口 (无需 HuggingFace Trainer 的轻量版本)
    # ================================================================

    def train_simple(
        self,
        train_data: list[dict],
        eval_data: Optional[list[dict]] = None,
    ) -> dict:
        """简化的训练接口 — 不依赖 HuggingFace.

        Args:
            train_data: 训练数据 [{"messages": [...], "text": "..."}, ...].
            eval_data: 评估数据.

        Returns:
            训练结果字典.
        """
        try:
            # 尝试完整训练
            from datasets import Dataset

            train_dataset = Dataset.from_list(train_data)
            eval_dataset = Dataset.from_list(eval_data) if eval_data else None

            self.train(train_dataset, eval_dataset)
            return {"status": "success", "checkpoint": self.config.sft_checkpoint_dir}

        except ImportError as e:
            logger.warning(
                "Full training unavailable (%s). "
                "Install transformers + peft + datasets for actual training.",
                e,
            )
            return {
                "status": "dry_run",
                "message": (
                    f"Training pipeline configured. "
                    f"Base model: {self.config.base_model}, "
                    f"LoRA rank: {self.config.lora_r}, "
                    f"Epochs: {self.config.num_epochs_sft}, "
                    f"Samples: {len(train_data)}. "
                    f"Install transformers + peft + datasets to run actual training."
                ),
                "config": {
                    "base_model": self.config.base_model,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "learning_rate": self.config.learning_rate,
                    "num_epochs": self.config.num_epochs_sft,
                    "max_seq_length": self.config.max_seq_length,
                },
            }
