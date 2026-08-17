# retrieval_layer/retrieval/hybrid.py
"""混合检索器 — Dense + Sparse 双路召回 + RRF 融合.

融合策略:
    RRF (Reciprocal Rank Fusion):
        score(doc) = Σ 1 / (k + rank_i(doc))

    其中:
        - k = 60 (默认，防止高排名过度影响)
        - rank_i(doc) = 文档在第 i 路检索结果中的排名 (1-indexed)
"""

import logging
from typing import Optional

from retrieval_layer.retrieval.dense import DenseRetriever
from retrieval_layer.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器 — 融合稠密 & 稀疏召回结果.

    两路召回:
        Route 1: Dense (BGE 语义) → 语义相似度
        Route 2: Sparse (FTS5 BM25) → 关键词匹配
        → RRF 融合 → Top-K

    Usage::

        hybrid = HybridRetriever(dense_retriever, sparse_retriever)
        results = hybrid.search("贵州茅台 市盈率 估值", top_k=10)
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    # ================================================================
    # 搜索
    # ================================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        where: Optional[dict] = None,
        weights: Optional[dict] = None,
    ) -> list[dict]:
        """混合检索.

        Args:
            query: 查询文本.
            top_k: 最终返回数量.
            dense_top_k: 稠密路召回数.
            sparse_top_k: 稀疏路召回数.
            where: 元数据过滤 (仅 dense 路).
            weights: {'dense': 1.0, 'sparse': 1.0} 各路权重.

        Returns:
            融合后的检索结果列表.
        """
        if not query.strip():
            return []

        weights = weights or {"dense": 1.0, "sparse": 1.0}

        # 两路并发召回 (这里顺序执行，后续可改并行)
        dense_results = self.dense.search(query, top_k=dense_top_k, where=where)
        sparse_results = self.sparse.search(query, top_k=sparse_top_k)

        # RRF 融合
        fused = self._rrf_fusion(
            dense_results, sparse_results,
            w_dense=weights["dense"],
            w_sparse=weights["sparse"],
        )

        # Top-K
        fused = fused[:top_k]

        logger.debug(
            "Hybrid search '%s': dense=%d, sparse=%d, fused=%d",
            query[:30],
            len(dense_results),
            len(sparse_results),
            len(fused),
        )
        return fused

    # ================================================================
    # 融合算法
    # ================================================================

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        w_dense: float = 1.0,
        w_sparse: float = 1.0,
    ) -> list[dict]:
        """RRF (Reciprocal Rank Fusion) 融合.

        对每个文档计算:
            score = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)

        如果文档只出现在一路中，另一路的 rank 视为无穷大 (贡献 0)。
        """
        # 收集所有 unique 文档
        doc_map: dict[str, dict] = {}  # id → merged doc info
        scores: dict[str, float] = {}

        # Dense 路
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_score = w_dense / (self.rrf_k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

        # Sparse 路
        for rank, doc in enumerate(sparse_results, start=1):
            doc_id = doc["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            rrf_score = w_sparse / (self.rrf_k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

        # 按融合分降序排列
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for doc_id in ranked_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = scores[doc_id]
            doc["source"] = "hybrid"

            # 标记出现在哪些路
            in_dense = doc_id in {d["id"] for d in dense_results}
            in_sparse = doc_id in {s["id"] for s in sparse_results}
            if in_dense and in_sparse:
                doc["match_type"] = "both"
            elif in_dense:
                doc["match_type"] = "dense_only"
            else:
                doc["match_type"] = "sparse_only"

            results.append(doc)

        return results
