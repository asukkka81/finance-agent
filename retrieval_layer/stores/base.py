# retrieval_layer/stores/base.py
"""存储抽象基类."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class BaseVectorStore(ABC):
    """向量数据库抽象基类.

    封装向量存储的 CRUD + 相似度检索。
    """

    @abstractmethod
    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """批量添加向量 & 文档.

        Args:
            ids: 唯一 ID 列表.
            vectors: 向量矩阵 (N, dim).
            documents: 文档文本列表.
            metadatas: 元数据列表 (可选).
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """向量相似度检索.

        Args:
            query_vector: 查询向量 (dim,).
            top_k: 返回数量.
            where: ChromaDB 风格的元数据过滤条件.

        Returns:
            [{'id': ..., 'document': ..., 'metadata': ..., 'score': ...}, ...]
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """按 ID 删除文档."""
        ...

    @abstractmethod
    def count(self) -> int:
        """返回文档总数."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空所有数据."""
        ...


class BaseTextStore(ABC):
    """全文检索引擎抽象基类."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """批量添加文档到全文索引."""
        ...

    @abstractmethod
    def search(
        self, query: str, top_k: int = 10
    ) -> list[dict]:
        """全文检索.

        Returns:
            [{'id': ..., 'document': ..., 'score': ...}, ...]
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """按 ID 删除."""
        ...

    @abstractmethod
    def count(self) -> int:
        """返回文档总数."""
        ...
