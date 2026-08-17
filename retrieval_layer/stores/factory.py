# retrieval_layer/stores/factory.py
"""存储工厂 — 创建向量存储 + 全文索引实例."""

import logging
from typing import Optional

from retrieval_layer.config import RetrievalConfig
from retrieval_layer.stores.base import BaseTextStore, BaseVectorStore
from retrieval_layer.stores.chroma_store import ChromaVectorStore
from retrieval_layer.stores.fts5_store import FTS5Index

logger = logging.getLogger(__name__)


class StoreFactory:
    """存储工厂.

    Usage::

        config = RetrievalConfig()
        factory = StoreFactory(config)
        vector_store = factory.create_vector_store()
        text_store = factory.create_text_store()
    """

    def __init__(self, config: RetrievalConfig):
        self._config = config
        self._vector_store: Optional[BaseVectorStore] = None
        self._text_store: Optional[BaseTextStore] = None

    def create_vector_store(
        self, collection_name: str = "finance_knowledge"
    ) -> BaseVectorStore:
        """创建或获取 ChromaDB 向量存储 (单例)."""
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore(
                path=self._config.chroma_path,
                collection_name=collection_name,
            )
        return self._vector_store

    def create_text_store(self) -> BaseTextStore:
        """创建或获取 FTS5 全文索引 (单例)."""
        if self._text_store is None:
            self._text_store = FTS5Index(db_path=self._config.fts5_db_path)
        return self._text_store

    @property
    def vector_store(self) -> Optional[BaseVectorStore]:
        return self._vector_store

    @property
    def text_store(self) -> Optional[BaseTextStore]:
        return self._text_store
