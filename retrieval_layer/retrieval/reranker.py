# retrieval_layer/retrieval/reranker.py
"""重排序器 — 用 Cross-Encoder 对候选结果精排.

提供两种 Reranker:
    1. BGE Reranker (FlagEmbedding): 轻量、针对中文金融优化
    2. (预留) LLM-based Reranker: 用 LLM 打分，更准但更慢
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Reranker:
    """BGE Cross-Encoder 重排序器.

    在混合检索的候选集中用 Cross-Encoder 重新打分，
    显著提升 Top-K 精度。

    Usage::

        reranker = Reranker("BAAI/bge-reranker-base")
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        top_k: int = 10,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None

        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                model_name,
                use_fp16=(device != "cpu"),
                device=device,
            )
            logger.info("Reranker loaded: %s on %s", model_name, device)
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed. "
                "Reranker will use fallback (identity rerank). "
                "Install with: pip install FlagEmbedding"
            )
        except Exception as e:
            logger.warning("Failed to load reranker %s: %s", model_name, e)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """对候选结果重排序.

        Args:
            query: 查询文本.
            candidates: 候选结果列表 (每个有 'document' 字段).
            top_k: 最终返回数量 (默认使用 self.top_k).

        Returns:
            重排序后的结果列表 (score 字段更新为 rerank 分数).
        """
        if top_k is None:
            top_k = self.top_k

        if not candidates:
            return []

        if not self.is_available:
            # 回退: 保持原序
            logger.debug("Reranker unavailable, returning original order")
            return candidates[:top_k]

        # 构建 query-document 对
        pairs = [[query, c.get("document", "")] for c in candidates]

        # Cross-Encoder 打分
        try:
            scores = self._model.compute_score(pairs, normalize=True)

            # 处理单结果情况
            if isinstance(scores, float):
                scores = [scores]

            # 更新分数 & 排序
            for i, candidate in enumerate(candidates):
                candidate["rerank_score"] = float(scores[i])
                candidate["original_score"] = candidate.get("score", 0.0)
                candidate["score"] = candidate["rerank_score"]
                candidate["source"] = candidate.get("source", "unknown") + "_reranked"

            # 按 rerank 分数降序
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        except Exception as e:
            logger.warning("Reranker compute failed: %s", e)

        return candidates[:top_k]

    def batch_rerank(
        self,
        queries: list[str],
        candidates_batch: list[list[dict]],
        top_k: Optional[int] = None,
    ) -> list[list[dict]]:
        """批量重排序."""
        return [
            self.rerank(q, c, top_k)
            for q, c in zip(queries, candidates_batch)
        ]
