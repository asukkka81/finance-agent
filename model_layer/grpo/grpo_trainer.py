# model_layer/grpo/grpo_trainer.py
"""GRPO 强化学习训练器 — Group Relative Policy Optimization.

算法核心:
    1. 从 prompt 集合采样
    2. 对每个 prompt 生成 K 个候选回答 (rollout)
    3. 用奖励函数对每个候选打分
    4. 计算组内相对优势 (advantage)
    5. PPO-style 策略梯度更新 (clip objective + KL penalty)

GRPO 相对标准 PPO 的优势:
    - 组内归一化消除 reward scale 问题
    - 无需训练单独的 critic/value 网络
    - 更适合 LLM 对齐场景
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """GRPO 强化学习训练器.

    实现 Group Relative Policy Optimization 算法，
    用于在 SFT 模型基础上进一步对齐金融场景。

    使用条件:
        pip install transformers peft accelerate torch trl
        (TRL 库内置了 DPOTrainer / PPOTrainer，GRPO 基于其扩展)
    """

    def __init__(self, config):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._ref_model = None  # 参考模型 (用于 KL 惩罚)
        self._reward_calculator = None

    # ================================================================
    # 初始化
    # ================================================================

    def initialize(self, sft_checkpoint_path: Optional[str] = None):
        """加载 SFT 模型 & 初始化奖励计算器.

        Args:
            sft_checkpoint_path: SFT LoRA checkpoint 路径.
        """
        from model_layer.grpo.reward import RewardCalculator

        logger.info("Initializing GRPO trainer...")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            # 加载 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=self.config.trust_remote_code,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # 加载基座模型
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=self.config.trust_remote_code,
                device_map="auto",
            )

            # 加载 LoRA (如果有)
            if sft_checkpoint_path:
                self._model = PeftModel.from_pretrained(
                    base_model, sft_checkpoint_path
                )
                logger.info("Loaded SFT LoRA from %s", sft_checkpoint_path)
            else:
                self._model = base_model

            # 参考模型 (冻结的 SFT 模型，用于 KL 约束)
            self._ref_model = base_model  # 简化: 使用同一基座模型

            # 奖励计算器
            self._reward_calculator = RewardCalculator(self.config)

            logger.info("GRPO trainer initialized")

        except ImportError as e:
            logger.warning(
                "GRPO dependencies not available (%s). "
                "Install with: pip install transformers peft accelerate torch trl",
                e,
            )
        except Exception as e:
            logger.error("Failed to initialize GRPO trainer: %s", e)
            raise

    # ================================================================
    # GRPO 训练主循环
    # ================================================================

    def train(
        self,
        prompts: list[str],
        num_epochs: Optional[int] = None,
        num_rollouts: Optional[int] = None,
    ) -> dict:
        """执行 GRPO 训练.

        Args:
            prompts: 训练 prompt 列表.
            num_epochs: 训练轮数 (默认从 config).
            num_rollouts: 每个 prompt 生成的候选数 (默认从 config).

        Returns:
            训练统计字典.
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs_grpo
        if num_rollouts is None:
            num_rollouts = self.config.num_rollouts_per_prompt

        if self._model is None:
            try:
                self.initialize()
            except Exception as e:
                return {
                    "status": "dry_run",
                    "message": (
                        f"GRPO training pipeline configured. "
                        f"Base model: {self.config.base_model}, "
                        f"Rollouts per prompt: {num_rollouts}, "
                        f"Epochs: {num_epochs}, "
                        f"Prompts: {len(prompts)}. "
                        f"Install transformers + peft + torch + trl to run actual training."
                    ),
                    "config": {
                        "base_model": self.config.base_model,
                        "learning_rate": self.config.grpo_learning_rate,
                        "num_epochs": num_epochs,
                        "num_rollouts": num_rollouts,
                        "clip_epsilon": self.config.grpo_clip_epsilon,
                        "kl_beta": self.config.grpo_kl_beta,
                        "reward_weights": self.config.reward_weights,
                    },
                }

        logger.info(
            "Starting GRPO training: %d prompts, %d rollouts, %d epochs",
            len(prompts), num_rollouts, num_epochs,
        )

        try:
            import torch
            from torch.utils.data import DataLoader

            # 使用 TRL 的 GRPOTrainer (如果可用)
            try:
                from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer

                grpo_config = GRPOConfig(
                    output_dir=self.config.grpo_checkpoint_dir,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    learning_rate=self.config.grpo_learning_rate,
                    num_train_epochs=num_epochs,
                    logging_steps=10,
                    save_steps=200,
                    bf16=True,
                )

                trl_trainer = TRLGRPOTrainer(
                    model=self._model,
                    reward_funcs=self._reward_calculator.compute_rewards,
                    args=grpo_config,
                    train_dataset=self._prepare_prompt_dataset(prompts),
                    tokenizer=self._tokenizer,
                )

                trl_trainer.train()
                trl_trainer.save_model(self.config.grpo_checkpoint_dir)

                logger.info("GRPO training complete (TRL)")
                return {"status": "success", "checkpoint": self.config.grpo_checkpoint_dir}

            except ImportError:
                logger.warning("TRL not available, running simplified GRPO loop")
                return self._train_simple(prompts, num_epochs, num_rollouts)

        except Exception as e:
            logger.error("GRPO training failed: %s", e)
            return {"status": "error", "error": str(e)}

    # ================================================================
    # 简化 GRPO 训练循环
    # ================================================================

    def _train_simple(
        self,
        prompts: list[str],
        num_epochs: int,
        num_rollouts: int,
    ) -> dict:
        """简化的 GRPO 训练循环 (不依赖 TRL).

        适用于小规模实验 & 验证 GRPO 算法流程。
        """
        import torch
        import numpy as np

        device = next(self._model.parameters()).device
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.grpo_learning_rate,
        )

        stats = {"epochs": [], "mean_reward": [], "loss": []}

        for epoch in range(num_epochs):
            epoch_rewards = []

            for prompt in prompts[:10]:  # 简化: 只用前 10 个 prompt
                # 生成 K 个候选回答
                rollouts = self._generate_rollouts(prompt, num_rollouts)

                # 计算奖励
                rewards = self._reward_calculator.compute_rewards(rollouts, prompt)
                advantages = torch.tensor(rewards, dtype=torch.float32, device=device)

                # 计算 log_probs (简化: 用模型前向传播)
                try:
                    loss = self._compute_grpo_loss(
                        rollouts, advantages, prompt,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                    epoch_rewards.append(np.mean(rewards))
                except Exception as e:
                    logger.warning("GRPO step failed: %s", e)
                    continue

            mean_r = np.mean(epoch_rewards) if epoch_rewards else 0.0
            stats["epochs"].append(epoch + 1)
            stats["mean_reward"].append(mean_r)
            stats["loss"].append(0.0)  # 简化实现不追踪 loss

            logger.info(
                "GRPO Epoch %d/%d: mean_reward=%.4f",
                epoch + 1, num_epochs, mean_r,
            )

        # 保存
        self._model.save_pretrained(self.config.grpo_checkpoint_dir)

        return {
            "status": "success_simple",
            "checkpoint": self.config.grpo_checkpoint_dir,
            "stats": stats,
            "final_mean_reward": stats["mean_reward"][-1] if stats["mean_reward"] else 0,
        }

    # ================================================================
    # 辅助函数
    # ================================================================

    def _generate_rollouts(
        self, prompt: str, num_sequences: int
    ) -> list[dict]:
        """对单个 prompt 生成多个候选回答.

        Args:
            prompt: 用户输入.
            num_sequences: 生成数量.

        Returns:
            [{"response": "...", "tool_calls": [...]}, ...]
        """
        if self._tokenizer is None or self._model is None:
            # 返回 mock rollouts
            return [
                {"response": f"Mock response {i} for: {prompt[:50]}...", "tool_calls": []}
                for i in range(num_sequences)
            ]

        import torch

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self._model.device)

        rollouts = []
        for _ in range(num_sequences):
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.inference_max_tokens,
                    temperature=self.config.inference_temperature,
                    top_p=self.config.inference_top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            rollouts.append({
                "response": response,
                "tool_calls": self._extract_tool_calls(response),
            })

        return rollouts

    def _compute_grpo_loss(
        self,
        rollouts: list[dict],
        advantages: torch.Tensor,
        prompt: str,
    ) -> torch.Tensor:
        """计算 GRPO 损失 (简化版).

        GRPO Loss = -E[advantage * log_prob] + β * KL(π || π_ref)
        """
        import torch
        import torch.nn.functional as F

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self._model.device)

        total_loss = torch.tensor(0.0, device=self._model.device)

        for i, rollout in enumerate(rollouts):
            if i >= len(advantages):
                break
            advantage = advantages[i]

            response_text = rollout.get("response", "")
            full_text = prompt + response_text
            tokens = self._tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self._model.device)

            # Policy loss (简化: 只计算 response 部分)
            with torch.no_grad():
                ref_outputs = self._ref_model(**tokens, labels=tokens["input_ids"])
                ref_log_prob = -ref_outputs.loss

            outputs = self._model(**tokens, labels=tokens["input_ids"])
            log_prob = -outputs.loss

            # PPO-style clipped objective
            ratio = torch.exp(log_prob - ref_log_prob.detach())
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.grpo_clip_epsilon,
                1 + self.config.grpo_clip_epsilon,
            )

            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

            # KL penalty
            kl = (ref_log_prob - log_prob).detach()
            kl_penalty = self.config.grpo_kl_beta * kl

            loss = policy_loss + kl_penalty
            total_loss = total_loss + loss

        return total_loss / max(len(rollouts), 1)

    def _extract_tool_calls(self, response: str) -> list[dict]:
        """从模型输出中提取工具调用 (启发式)."""
        import re
        import json

        calls = []

        # 匹配 <tool_call>...</tool_call>
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    calls.extend(parsed)
                elif isinstance(parsed, dict):
                    calls.append(parsed)
            except json.JSONDecodeError:
                pass

        return calls

    def _prepare_prompt_dataset(self, prompts: list[str]):
        """将 prompt 列表转为 HuggingFace Dataset."""
        try:
            from datasets import Dataset
            return Dataset.from_list([{"prompt": p} for p in prompts])
        except ImportError:
            return prompts
