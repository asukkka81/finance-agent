# retrieval_layer/embeddings/bge_embedder.py
"""BGE Embedding 模型封装.

BGE (BAAI General Embedding) 是当前中文语义检索的 SOTA 模型。
金融领域可用 BAAI/bge-small-zh-v1.5 (轻量) 或 bge-large-zh-v1.5 (精度).

BGE 对 query 和 passage 使用不同的 instruction:
    - query: "为这个句子生成表示以用于检索相关文章："
    - passage: "" (空)
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval_layer.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)

# BGE 中文 query instruction
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BGEEmbedder(BaseEmbedder):
    """BGE Embedding 模型.

    Usage::

        embedder = BGEEmbedder("BAAI/bge-small-zh-v1.5")
        vectors = embedder.encode(["文本1", "文本2"])
        query_vec = embedder.encode_query("查询文本")
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self._model_name = model_name
        self._device = device
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info("Loading BGE model: %s on %s ...", model_name, device)
        self._model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "BGE model loaded: %s (dim=%d)", model_name, self._dimension
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self, texts: list[str], show_progress: bool = False
    ) -> np.ndarray:
        """批量编码文档 (不加 instruction).

        Args:
            texts: 待编码的文档文本列表.
            show_progress: 是否显示进度条.

        Returns:
            shape=(len(texts), dimension) 的 float32 数组.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询 (加 BGE query instruction).

        查询文本会自动添加 instruction 前缀。
        """
        instructed = f"{BGE_QUERY_INSTRUCTION}{query}"
        embedding = self._model.encode(
            [instructed],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """批量编码查询 (加 instruction)."""
        instructed = [f"{BGE_QUERY_INSTRUCTION}{q}" for q in queries]
        return self.encode(instructed)

    def get_model(self) -> SentenceTransformer:
        """返回底层 SentenceTransformer 模型 (用于 Reranker 等)."""
        return self._model
