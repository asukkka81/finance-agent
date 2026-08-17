# model_layer/config.py
"""模型层全局配置 — LoRA SFT + GRPO + 推理."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型训练 & 推理配置.

    Attributes:
        base_model: 基座模型名称 (Qwen3-14B).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout.
        learning_rate: SFT 学习率.
        grpo_learning_rate: GRPO 学习率.
        max_seq_length: 最大序列长度.
        per_device_batch_size: 每设备 batch size.
        gradient_accumulation_steps: 梯度累积步数.
        num_epochs_sft: SFT 训练轮数.
        num_epochs_grpo: GRPO 训练轮数.
    """

    # ================================================================
    # 基座模型
    # ================================================================
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"

    # ================================================================
    # LoRA 参数
    # ================================================================
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # 额外训练模块 (embedding / lm_head 等)
    modules_to_save: list[str] = field(default_factory=list)

    # ================================================================
    # SFT 训练
    # ================================================================
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 4096
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs_sft: int = 3
    max_steps_sft: int = -1  # -1 = 使用 epochs
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50

    # ================================================================
    # GRPO 训练
    # ================================================================
    grpo_learning_rate: float = 5e-6
    num_epochs_grpo: int = 2
    num_rollouts_per_prompt: int = 4     # 每个 prompt 生成的候选数 (组大小)
    grpo_clip_epsilon: float = 0.2       # PPO clip 范围
    grpo_kl_beta: float = 0.01           # KL 散度惩罚系数
    reward_weights: dict = field(default_factory=lambda: {
        "format_compliance": 0.20,   # 格式合规
        "tool_scheduling": 0.30,     # 工具调度质量
        "code_quality": 0.15,        # 代码质量 (沙箱)
        "answer_quality": 0.35,      # 答案质量
    })

    # ================================================================
    # 推理
    # ================================================================
    vllm_tensor_parallel: int = 1
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_model_len: int = 8192
    inference_temperature: float = 0.1
    inference_top_p: float = 0.9
    inference_max_tokens: int = 2048

    # ================================================================
    # 数据
    # ================================================================
    train_data_path: str = "data/training/sft_train.jsonl"
    eval_data_path: str = "data/training/sft_eval.jsonl"
    grpo_prompt_path: str = "data/training/grpo_prompts.jsonl"
    output_dir: str = "outputs/"
    sft_checkpoint_dir: str = "outputs/sft_checkpoint"
    grpo_checkpoint_dir: str = "outputs/grpo_checkpoint"
    merged_model_dir: str = "outputs/merged_model"

    # ================================================================
    # 奖励函数维度定义
    # ================================================================
    reward_dimensions: dict = field(default_factory=lambda: {
        "format_compliance": {
            "description": "回答格式是否符合规范 (风险提示、数据引用、结构清晰)",
            "weight": 0.20,
        },
        "tool_scheduling": {
            "description": "工具调用是否合理 (顺序正确、无冗余、参数准确)",
            "weight": 0.30,
        },
        "code_quality": {
            "description": "Python 代码是否安全、正确、高效 (沙箱执行结果)",
            "weight": 0.15,
        },
        "answer_quality": {
            "description": "答案是否准确、完整、逻辑自洽",
            "weight": 0.35,
        },
    })
