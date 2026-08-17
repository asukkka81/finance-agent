# retrieval_layer/stores/chroma_store.py
"""ChromaDB 向量存储."""

import logging
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from retrieval_layer.stores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB 向量存储封装.

    特点:
        - 本地持久化 (非内存模式)
        - 支持元数据过滤
        - 自动管理 Collection

    Usage::

        store = ChromaVectorStore(path="data/chroma_db")
        store.add(ids=["doc1"], vectors=emb, documents=["text..."])
        results = store.search(query_vector, top_k=10)
    """

    def __init__(
        self,
        path: str = "data/chroma_db",
        collection_name: str = "finance_knowledge",
        distance_metric: str = "cosine",
    ):
        self.path = path
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )

        logger.info(
            "ChromaDB initialized: %s/%s (%d docs)",
            path, collection_name, self._collection.count(),
        )

    # ================================================================
    # CRUD
    # ================================================================

    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """批量添加向量 & 文档."""
        if not ids:
            return

        # ChromaDB 要求向量是 list[list[float]]
        vectors_list = vectors.tolist() if isinstance(vectors, np.ndarray) else vectors

        self._collection.add(
            ids=ids,
            embeddings=vectors_list,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("Added %d documents to ChromaDB", len(ids))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """向量相似度检索.

        Returns:
            [
                {
                    'id': str,
                    'document': str,
                    'metadata': dict,
                    'score': float,  # 余弦距离 (越小越相似，0=完全相同)
                },
                ...
            ]
        """
        vec_list = (
            query_vector.tolist()
            if isinstance(query_vector, np.ndarray)
            else query_vector
        )

        results = self._collection.query(
            query_embeddings=[vec_list],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # 标准化输出格式
        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append({
                    "id": doc_id,
                    "document": (
                        results["documents"][0][i]
                        if results["documents"] else ""
                    ),
                    "metadata": (
                        results["metadatas"][0][i]
                        if results["metadatas"] else {}
                    ),
                    "score": (
                        1.0 - results["distances"][0][i]
                        if results["distances"] else 0.0
                    ),  # 转为相似度: 1.0 = 完全相同
                })

        return output

    def delete(self, ids: list[str]) -> None:
        """按 ID 删除文档."""
        if ids:
            self._collection.delete(ids=ids)

    def count(self) -> int:
        """返回文档总数."""
        return self._collection.count()

    def clear(self) -> None:
        """清空 Collection (删除后重建)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection cleared: %s", self.collection_name)

    # ================================================================
    # 管理
    # ================================================================

    def get_collection_stats(self) -> dict:
        """获取 Collection 统计信息."""
        return {
            "name": self.collection_name,
            "count": self._collection.count(),
            "path": self.path,
        }

    def list_collections(self) -> list[str]:
        """列出所有 Collection."""
        return [c.name for c in self._client.list_collections()]
