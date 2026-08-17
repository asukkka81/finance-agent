# retrieval_layer/indexer/indexer.py
"""索引器 — 完整的入索引流水线: 加载 → 分块 → 向量化 → 双写存储."""

import logging
from typing import Optional

import numpy as np

from retrieval_layer.config import RetrievalConfig
from retrieval_layer.embeddings.base import BaseEmbedder
from retrieval_layer.indexer.loader import Document, DocumentLoader
from retrieval_layer.indexer.splitter import TextSplitter
from retrieval_layer.stores.base import BaseTextStore, BaseVectorStore

logger = logging.getLogger(__name__)


class Indexer:
    """索引器 — 文档入索引的编排器.

    流水线: 加载 → 分块 → 向量化 → 写入 ChromaDB + FTS5.

    Usage::

        embedder = BGEEmbedder()
        vector_store = ChromaVectorStore(...)
        text_store = FTS5Index(...)

        indexer = Indexer(embedder, vector_store, text_store)
        indexer.index_directory("data/documents/")
        indexer.index_stock_profiles(stocks_from_db)
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        text_store: BaseTextStore,
        config: Optional[RetrievalConfig] = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.text_store = text_store
        self.config = config or RetrievalConfig()

        self.loader = DocumentLoader()
        self.splitter = TextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    # ================================================================
    # 索引方法
    # ================================================================

    def index_documents(
        self,
        documents: list[Document],
        batch_size: int = 32,
    ) -> int:
        """索引 Document 列表.

        流程:
            1. 分块
            2. 批量向量化
            3. 写入 ChromaDB (向量检索)
            4. 写入 FTS5 (全文检索)

        Returns:
            int: 实际索引的 chunk 数.
        """
        if not documents:
            return 0

        # Step 1: 分块
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            logger.warning("No chunks generated from %d documents", len(documents))
            return 0

        logger.info(
            "Indexing %d documents → %d chunks", len(documents), len(chunks)
        )

        # Step 2-4: 批量处理
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            total += self._index_batch(batch)

        logger.info(
            "Indexing complete: %d chunks written (Chroma: %d, FTS5: %d)",
            total,
            self.vector_store.count(),
            self.text_store.count(),
        )
        return total

    def index_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        metadata: Optional[dict] = None,
    ) -> int:
        """索引目录中的所有文档.

        Args:
            dir_path: 目录路径.
            recursive: 是否递归.
            metadata: 全局元数据.

        Returns:
            int: 索引的 chunk 数.
        """
        documents = self.loader.load_directory(
            dir_path, recursive=recursive, metadata=metadata,
        )
        return self.index_documents(documents)

    def index_stock_profiles(self, stocks: list) -> int:
        """索引股票基本信息 (从 data_layer 导入).

        Args:
            stocks: list[Stock] 列表.

        Returns:
            int: 索引的 chunk 数.
        """
        documents = self.loader.load_stock_profiles(stocks)
        return self.index_documents(documents)

    def index_dicts(
        self,
        records: list[dict],
        content_key: str = "content",
    ) -> int:
        """索引字典列表."""
        documents = self.loader.load_from_dicts(records, content_key)
        return self.index_documents(documents)

    # ================================================================
    # 内部
    # ================================================================

    def _index_batch(self, chunks: list[Document]) -> int:
        """索引一批 chunk.

        返回实际写入的数量。
        """
        ids = [c.id for c in chunks]
        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # 向量化
        try:
            vectors = self.embedder.encode(texts)
        except Exception as e:
            logger.error("Embedding failed for batch: %s", e)
            return 0

        # 双写: ChromaDB + FTS5
        try:
            self.vector_store.add(ids, vectors, texts, metadatas)
        except Exception as e:
            logger.error("ChromaDB write failed: %s", e)
            raise

        try:
            self.text_store.add(ids, texts, metadatas)
        except Exception as e:
            logger.error("FTS5 write failed: %s", e)

        return len(chunks)
