# retrieval_layer/embeddings/factory.py
"""Embedding 模型工厂 — 按名称创建 & 缓存."""

import logging
from typing import Optional

from retrieval_layer.embeddings.base import BaseEmbedder
from retrieval_layer.embeddings.bge_embedder import BGEEmbedder

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Embedding 模型工厂 (单例模式).

    Usage::

        factory = EmbeddingFactory()
        embedder = factory.get("BAAI/bge-small-zh-v1.5")
        query_vec = embedder.encode_query("什么是ROE")
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self._device = device
        self._batch_size = batch_size
        self._cache: dict[str, BaseEmbedder] = {}

    def get(self, model_name: Optional[str] = None) -> BaseEmbedder:
        """获取或创建 Embedding 模型.

        Args:
            model_name: 模型名称. None 时使用默认 BGE 中文模型.

        Returns:
            BaseEmbedder 实例 (缓存的).
        """
        if model_name is None:
            model_name = "BAAI/bge-small-zh-v1.5"

        if model_name not in self._cache:
            self._cache[model_name] = self._create(model_name)

        return self._cache[model_name]

    def _create(self, model_name: str) -> BaseEmbedder:
        """创建 Embedder 实例."""
        # BGE 系列
        if "bge" in model_name.lower():
            return BGEEmbedder(
                model_name=model_name,
                device=self._device,
                batch_size=self._batch_size,
            )

        # GME 系列 (多模态，后续扩展)
        if "gme" in model_name.lower():
            raise NotImplementedError(
                "GME multimodal embedding not yet supported"
            )

        # 兜底: 尝试用 SentenceTransformer 加载
        logger.warning(
            "Unknown model type '%s', trying SentenceTransformer...", model_name
        )
        return BGEEmbedder(
            model_name=model_name,
            device=self._device,
            batch_size=self._batch_size,
        )

    def clear_cache(self) -> None:
        """清除模型缓存 (释放 GPU 内存)."""
        self._cache.clear()

    @property
    def loaded_models(self) -> list[str]:
        """列出已加载的模型名称."""
        return list(self._cache.keys())
