# retrieval_layer/embeddings/base.py
"""Embedding 模型抽象基类."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Embedding 模型基类.

    所有 embedding 模型 (BGE, GME, ...) 实现此接口。
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回模型名称."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度."""
        ...

    @abstractmethod
    def encode(
        self, texts: list[str], show_progress: bool = False
    ) -> np.ndarray:
        """将文本列表编码为向量矩阵.

        Args:
            texts: 待编码的文本列表.
            show_progress: 是否显示进度条.

        Returns:
            shape=(len(texts), dimension) 的 numpy 数组.
        """
        ...

    @abstractmethod
    def encode_query(self, query: str) -> np.ndarray:
        """将查询文本编码为向量 (单条).

        与 encode() 的区别: 某些模型对 query 和 document 使用不同的
        instruction/prompt，此方法自动处理。

        Args:
            query: 查询字符串.

        Returns:
            shape=(dimension,) 的 numpy 数组.
        """
        ...

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """批量编码查询文本."""
        return self.encode(queries)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度."""
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        )
