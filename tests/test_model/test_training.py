# tests/test_model/test_training.py
"""训练流程测试 (dry run)."""

import pytest

from model_layer.config import ModelConfig
from model_layer.data.dataset import FinanceDataset
from model_layer.data.synthesizer import DataSynthesizer


class TestSFTTrainer:
    """测试 SFT 训练器配置."""

    def test_lora_config(self):
        """LoRA 配置参数验证."""
        config = ModelConfig(
            base_model="Qwen/Qwen2.5-14B-Instruct",
            num_epochs_sft=1,
            lora_r=64,
        )

        from model_layer.sft.lora_config import get_lora_config, estimate_trainable_params

        # LoRA 配置
        lora_cfg = get_lora_config(config)
        assert lora_cfg["r"] == config.lora_r
        assert "q_proj" in lora_cfg["target_modules"]

        # 参数估算
        params = estimate_trainable_params(config.base_model, lora_cfg)
        assert params["total_params"] > 0
        assert params["trainable_params"] > 0
        assert 0 < params["ratio"] < 10  # LoRA 参数量 < 10%

    def test_simple_train_dry_run(self):
        """SFT 简化训练 dry run."""
        config = ModelConfig(
            base_model="Qwen/Qwen2.5-14B-Instruct",
            num_epochs_sft=1,
        )

        from model_layer.sft.trainer import SFTTrainer
        from model_layer.data.synthesizer import DataSynthesizer
        from model_layer.data.dataset import FinanceDataset

        trainer = SFTTrainer(config)
        synth = DataSynthesizer(seed=42)
        dataset = FinanceDataset()
        dataset.load_from_synthesizer(synth, {"stock_price": 5, "knowledge_qa": 5})

        train_data = dataset.to_huggingface()
        result = trainer.train_simple(train_data)

        assert result["status"] in ("dry_run", "success")
        if "config" in result:
            assert result["config"]["base_model"] == config.base_model


class TestGRPOTrainer:
    """测试 GRPO 训练器 (dry run)."""

    def test_config_validation(self):
        """GRPO 配置验证."""
        config = ModelConfig()
        assert config.num_rollouts_per_prompt > 0
        assert config.grpo_learning_rate > 0
        assert sum(config.reward_weights.values()) > 0.99

    def test_reward_weights_sum(self):
        """奖励权重应接近 1.0."""
        config = ModelConfig()
        total = sum(config.reward_weights.values())
        assert 0.99 < total < 1.01


class TestLoraConfig:
    """测试 LoRA 配置."""

    def test_suggest_batch_size(self):
        from model_layer.sft.lora_config import suggest_batch_size

        # 80GB GPU → 较大 batch
        assert suggest_batch_size(80, 14) >= 4
        # 24GB GPU → 较小 batch
        assert suggest_batch_size(24, 14) <= 2
        # 16GB GPU → 最小
        assert suggest_batch_size(16, 14) == 1
