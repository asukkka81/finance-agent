# retrieval_layer/retrieval/sparse.py
"""稀疏检索器 — 基于 FTS5 的 BM25 关键词检索."""

import logging

from retrieval_layer.stores.base import BaseTextStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """全文关键词检索器.

    基于 SQLite FTS5 的 BM25 算法，精确匹配关键词。

    Usage::

        sparse = SparseRetriever(text_store)
        results = sparse.search("ROE 市盈率", top_k=10)
    """

    def __init__(self, text_store: BaseTextStore):
        self.text_store = text_store

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """全文检索.

        注意: FTS5 的中文分词是逐字符的，所以中文查询更接近 n-gram 匹配。

        Args:
            query: 搜索关键词.
            top_k: 返回数量.

        Returns:
            [{'id': ..., 'document': ..., 'score': ..., 'metadata': ..., 'source': 'sparse'}, ...]
        """
        if not query.strip():
            return []

        results = self.text_store.search(query, top_k=top_k)

        # 标记来源
        for r in results:
            r["source"] = "sparse"

        logger.debug(
            "Sparse search '%s': %d results",
            query[:30], len(results),
        )
        return results

    def phrase_search(self, phrase: str, top_k: int = 20) -> list[dict]:
        """精确短语搜索."""
        results = self.text_store.search(f'"{phrase}"', top_k=top_k)
        for r in results:
            r["source"] = "sparse_phrase"
        return results
