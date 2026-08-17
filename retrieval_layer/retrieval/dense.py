# retrieval_layer/retrieval/dense.py
"""稠密检索器 — 基于语义向量的相似度搜索."""

import logging
from typing import Optional

import numpy as np

from retrieval_layer.embeddings.base import BaseEmbedder
from retrieval_layer.stores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """向量语义检索器.

    流程: query → embedder.encode_query() → vector_store.search()

    Usage::

        dense = DenseRetriever(embedder, vector_store)
        results = dense.search("什么是市盈率", top_k=10)
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """语义检索.

        Args:
            query: 查询文本.
            top_k: 返回数量.
            where: 元数据过滤条件.

        Returns:
            [{'id': ..., 'document': ..., 'score': ..., 'metadata': ..., 'source': 'dense'}, ...]
        """
        if not query.strip():
            return []

        # Query → vector
        query_vector = self.embedder.encode_query(query)

        # Vector → search
        results = self.vector_store.search(
            query_vector, top_k=top_k, where=where,
        )

        # 标记来源
        for r in results:
            r["source"] = "dense"

        logger.debug(
            "Dense search '%s': %d results (top score: %.4f)",
            query[:30], len(results),
            results[0]["score"] if results else 0.0,
        )
        return results

    def batch_search(
        self,
        queries: list[str],
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[list[dict]]:
        """批量语义检索."""
        query_vectors = self.embedder.encode_queries(queries)

        all_results = []
        for i, qv in enumerate(query_vectors):
            results = self.vector_store.search(qv, top_k=top_k, where=where)
            for r in results:
                r["source"] = "dense"
            all_results.append(results)

        return all_results
