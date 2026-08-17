# retrieval_layer/pipelines/search.py
"""搜索流水线 — 面向 Agent 的统一检索入口.

完整检索流程:
    Query → Hybrid (Dense + Sparse) → RRF 融合 → Reranker → 过滤 → 结果
"""

import logging
from typing import Optional

from retrieval_layer.config import RetrievalConfig
from retrieval_layer.embeddings.base import BaseEmbedder
from retrieval_layer.retrieval.dense import DenseRetriever
from retrieval_layer.retrieval.hybrid import HybridRetriever
from retrieval_layer.retrieval.reranker import Reranker
from retrieval_layer.retrieval.sparse import SparseRetriever
from retrieval_layer.stores.base import BaseTextStore, BaseVectorStore

logger = logging.getLogger(__name__)


class SearchPipeline:
    """统一搜索流水线 — 面向 Agent 的检索 API.

    这是检索层的顶层入口，Agent 只需调用这一个接口。

    流程:
        1. Query pre-processing (规范化)
        2. Hybrid retrieval (Dense + Sparse → RRF)
        3. Reranking (Cross-Encoder 精排)
        4. Post-processing (时效降级 / 权限过滤 / 去重)
        5. 返回 Top-K

    Usage::

        pipeline = SearchPipeline(embedder, vector_store, text_store, config)
        results = pipeline.search("请分析贵州茅台的估值水平", top_k=5)

        # 带过滤
        results = pipeline.search(
            "新能源行业前景",
            top_k=5,
            filter_doc_type="research_report",
        )
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        text_store: BaseTextStore,
        config: Optional[RetrievalConfig] = None,
    ):
        self.config = config or RetrievalConfig()
        self.embedder = embedder

        # 初始化检索器
        self.dense = DenseRetriever(embedder, vector_store)
        self.sparse = SparseRetriever(text_store)
        self.hybrid = HybridRetriever(
            self.dense, self.sparse, rrf_k=self.config.rrf_k,
        )
        self.reranker = Reranker(
            model_name=self.config.reranker_model,
            top_k=self.config.top_k_final,
        )

    # ================================================================
    # 主要搜索接口
    # ================================================================

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_doc_type: Optional[str] = None,
        filter_market: Optional[str] = None,
        filter_authority_min: Optional[float] = None,
        enable_rerank: Optional[bool] = None,
        **kwargs,
    ) -> list[dict]:
        """统一搜索接口 — Agent 只需调用此方法.

        Args:
            query: 自然语言查询，e.g. "贵州茅台的ROE变化趋势".
            top_k: 最终返回数量 (默认从 config).
            filter_doc_type: 按文档类型过滤 ('research_report' / 'policy' / ...).
            filter_market: 按市场过滤 ('US' / 'CN').
            filter_authority_min: 最低权威度阈值 (0.0~1.0).
            enable_rerank: 是否启用重排序 (默认 True).

        Returns:
            [
                {
                    'id': str,
                    'document': str,       # 文档内容
                    'score': float,        # 最终分数 (0~1)
                    'metadata': dict,      # 元数据
                    'source': str,         # 'hybrid_reranked' | 'hybrid'
                    'match_type': str,     # 'both' | 'dense_only' | 'sparse_only'
                },
                ...
            ]
        """
        if top_k is None:
            top_k = self.config.top_k_final
        if enable_rerank is None:
            enable_rerank = self.config.enable_reranker
        if filter_authority_min is None:
            filter_authority_min = self.config.min_authority_score

        # ---- Step 1: 构建过滤条件 ----
        where = self._build_filter(filter_doc_type, filter_market)

        # ---- Step 2: Hybrid 检索 ----
        candidates = self.hybrid.search(
            query,
            top_k=top_k * 3,  # 多召回一些给 reranker
            dense_top_k=self.config.top_k_dense,
            sparse_top_k=self.config.top_k_sparse,
            where=where,
        )

        if not candidates:
            logger.info("No results for query: %s", query[:50])
            return []

        # ---- Step 3: Rerank ----
        if enable_rerank and self.reranker.is_available:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k * 2)
        else:
            candidates = candidates[:top_k * 2]

        # ---- Step 4: 后处理过滤 ----
        results = self._post_filter(candidates, filter_authority_min)

        # ---- Step 5: Top-K ----
        results = results[:top_k]

        logger.info(
            "Search '%s': %d candidates → %d results (top score: %.4f)",
            query[:50],
            len(candidates),
            len(results),
            results[0]["score"] if results else 0.0,
        )
        return results

    # ================================================================
    # 快捷搜索方法
    # ================================================================

    def search_dense_only(
        self, query: str, top_k: int = 10, **kwargs
    ) -> list[dict]:
        """纯语义检索 (不走混合)."""
        return self.dense.search(query, top_k=top_k)

    def search_sparse_only(
        self, query: str, top_k: int = 10
    ) -> list[dict]:
        """纯关键词检索."""
        return self.sparse.search(query, top_k=top_k)

    def search_stocks(
        self, query: str, market: Optional[str] = None, top_k: int = 5
    ) -> list[dict]:
        """搜索股票信息."""
        return self.search(
            query, top_k=top_k,
            filter_doc_type="stock_profile",
            filter_market=market,
        )

    def search_research(
        self, query: str, top_k: int = 5
    ) -> list[dict]:
        """搜索研报内容."""
        return self.search(
            query, top_k=top_k,
            filter_doc_type="research_report",
        )

    # ================================================================
    # 内部
    # ================================================================

    def _build_filter(
        self,
        doc_type: Optional[str] = None,
        market: Optional[str] = None,
    ) -> Optional[dict]:
        """构建 ChromaDB where 过滤条件."""
        conditions = []

        if doc_type:
            conditions.append({"doc_type": doc_type})

        if market:
            conditions.append({"market": market})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _post_filter(
        self,
        candidates: list[dict],
        min_authority: float = 0.0,
    ) -> list[dict]:
        """后处理过滤: 去重 (按内容相似度) + 权威度过滤 + 时效降级.

        Returns:
            过滤后的结果列表.
        """
        filtered = []

        # 权威度过滤
        for c in candidates:
            authority = c.get("metadata", {}).get("authority", 0.5)
            if isinstance(authority, str):
                try:
                    authority = float(authority)
                except ValueError:
                    authority = 0.5
            if authority < min_authority:
                continue
            filtered.append(c)

        # 简单内容去重 (前100字符相同视为重复)
        seen_contents = set()
        deduped = []
        for c in filtered:
            content_fingerprint = c.get("document", "")[:100].strip()
            if content_fingerprint and content_fingerprint in seen_contents:
                continue
            seen_contents.add(content_fingerprint)
            deduped.append(c)

        return deduped
