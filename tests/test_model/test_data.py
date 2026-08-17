# tests/test_model/test_data.py
"""训练数据模块测试."""

import json
import tempfile
import os

import pytest

from model_layer.config import ModelConfig
from model_layer.data.format import DataFormatter, TrainingExample, ConversationTurn
from model_layer.data.dataset import FinanceDataset


class TestDataFormatter:
    """测试数据格式化器."""

    def test_format_basic_conversation(self):
        """基本对话格式化."""
        messages = [
            {"role": "system", "content": "你是金融助手"},
            {"role": "user", "content": "什么是PE？"},
            {"role": "assistant", "content": "PE是市盈率..."},
        ]
        result = DataFormatter.format_conversation(messages)
        assert "<|im_start|>system" in result
        assert "你是金融助手" in result
        assert "什么是PE？" in result
        assert "PE是市盈率" in result
        assert result.endswith("<|im_start|>assistant\n")

    def test_format_with_tool_calls(self):
        """带工具调用的对话."""
        messages = [
            {"role": "system", "content": "你是金融助手"},
            {"role": "user", "content": "查询茅台股价"},
            {"role": "assistant", "content": "我需要查询行情",
             "tool_calls": [{"name": "get_stock_price",
                            "input": {"symbol": "600519"}}]},
        ]
        result = DataFormatter.format_conversation(messages)
        assert "<tool_call>" in result
        assert "get_stock_price" in result
        assert "600519" in result

    def test_format_tool_results(self):
        """工具结果格式化."""
        results = [
            {"tool_name": "get_stock_price", "success": True,
             "data": {"close": "1850.50"}},
        ]
        formatted = DataFormatter._format_tool_results(results)
        assert "get_stock_price" in formatted
        assert "1850.50" in formatted

    def test_validate_example_valid(self):
        """有效样本应通过校验."""
        example = TrainingExample(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "query"},
                {"role": "assistant", "content": "answer"},
            ],
        )
        assert DataFormatter.validate_example(example)

    def test_validate_example_no_user(self):
        """无 user 消息的样本不应通过."""
        example = TrainingExample(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "answer"},
            ],
        )
        assert not DataFormatter.validate_example(example)

    def test_messages_to_sharegpt(self):
        """ShareGPT 格式转换."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = DataFormatter.messages_to_sharegpt(messages)
        assert result["conversations"][0]["from"] == "system"
        assert result["conversations"][1]["from"] == "human"
        assert result["conversations"][2]["from"] == "gpt"

    def test_validate_dataset(self):
        """批量校验."""
        examples = [
            TrainingExample(
                messages=[
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ],
                metadata={"intent_type": "stock_price"},
            ),
            TrainingExample(
                messages=[
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ],
                metadata={"intent_type": "knowledge_qa"},
            ),
            TrainingExample(
                messages=[{"role": "user", "content": "q3"}],  # 无 assistant
            ),
        ]
        stats = DataFormatter.validate_dataset(examples)
        assert stats["total"] == 3
        assert stats["valid"] == 2
        assert stats["invalid"] == 1
        assert stats["intent_distribution"]["stock_price"] == 1


class TestFinanceDataset:
    """测试数据集."""

    @pytest.fixture
    def sample_data(self):
        """创建示例数据集."""
        dataset = FinanceDataset()
        dataset.examples = [
            TrainingExample(
                messages=[
                    {"role": "user", "content": "查询茅台股价"},
                    {"role": "assistant", "content": "茅台最新收盘价..."},
                ],
                metadata={"intent_type": "stock_price", "symbol": "600519"},
            ),
            TrainingExample(
                messages=[
                    {"role": "user", "content": "什么是ROE？"},
                    {"role": "assistant", "content": "ROE是净资产收益率..."},
                ],
                metadata={"intent_type": "knowledge_qa"},
            ),
        ]
        return dataset

    def test_len(self, sample_data):
        assert len(sample_data) == 2

    def test_getitem(self, sample_data):
        assert sample_data[0].metadata["intent_type"] == "stock_price"

    def test_to_huggingface(self, sample_data):
        hf = sample_data.to_huggingface()
        assert len(hf) == 2
        assert "messages" in hf[0]
        assert "text" in hf[0]

    def test_train_test_split(self, sample_data):
        train, test = sample_data.train_test_split(test_ratio=0.5, seed=42)
        assert len(train) == 1
        assert len(test) == 1

    def test_jsonl_roundtrip(self, sample_data):
        """保存再加载应保持数据一致."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            temp_path = f.name

        try:
            sample_data.to_jsonl(temp_path)
            loaded = FinanceDataset().load_jsonl(temp_path)
            assert len(loaded) == len(sample_data)
            assert loaded[0].metadata["intent_type"] == sample_data[0].metadata["intent_type"]
        finally:
            os.unlink(temp_path)


class TestDataSynthesizer:
    """测试数据合成器."""

    def test_generate_stock_price(self):
        from model_layer.data.synthesizer import DataSynthesizer
        synth = DataSynthesizer(seed=42)
        examples = synth.generate_stock_price_examples(count=10)
        assert len(examples) == 10
        for ex in examples:
            assert DataFormatter.validate_example(ex)
            assert ex.metadata["intent_type"] == "stock_price"

    def test_generate_knowledge_qa(self):
        from model_layer.data.synthesizer import DataSynthesizer
        synth = DataSynthesizer(seed=42)
        examples = synth.generate_knowledge_qa_examples(count=10)
        assert len(examples) == 10
        for ex in examples:
            assert DataFormatter.validate_example(ex)
            assert ex.metadata["intent_type"] == "knowledge_qa"

    def test_generate_tool_chain(self):
        from model_layer.data.synthesizer import DataSynthesizer
        synth = DataSynthesizer(seed=42)
        examples = synth.generate_tool_chain_examples(count=10)
        assert len(examples) == 10
        # 工具链样本包含多轮 tool calls
        has_tool_calls = any(
            "tool_calls" in msg
            for ex in examples
            for msg in ex.messages
        )
        assert has_tool_calls

    def test_generate_sandbox(self):
        from model_layer.data.synthesizer import DataSynthesizer
        synth = DataSynthesizer(seed=42)
        examples = synth.generate_sandbox_examples(count=10)
        assert len(examples) == 10
        # 沙箱样本应包含沙箱代码
        has_sandbox = any(
            "execute_python" in str(ex.messages) or "code" in str(ex.metadata)
            for ex in examples
        )
        assert has_sandbox

    def test_generate_all(self):
        from model_layer.data.synthesizer import DataSynthesizer
        synth = DataSynthesizer(seed=42)
        counts = {"stock_price": 5, "knowledge_qa": 5,
                   "tool_chain": 3, "sandbox": 3}
        examples = synth.generate_all(counts)
        assert len(examples) == 16

    def test_save_to_file(self):
        from model_layer.data.synthesizer import DataSynthesizer
        import os
        synth = DataSynthesizer(seed=42)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            temp_path = f.name

        try:
            # generate_all populates self._examples for save_to_file
            synth.generate_all({"stock_price": 3, "knowledge_qa": 2,
                                "tool_chain": 1, "sandbox": 1})
            synth.save_to_file(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            os.unlink(temp_path)
