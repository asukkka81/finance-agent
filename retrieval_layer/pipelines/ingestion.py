# retrieval_layer/pipelines/ingestion.py
"""数据摄入流水线 — 将 data_layer 数据导入检索系统."""

import logging
from typing import Optional

from retrieval_layer.config import RetrievalConfig
from retrieval_layer.embeddings.base import BaseEmbedder
from retrieval_layer.indexer.indexer import Indexer
from retrieval_layer.stores.base import BaseTextStore, BaseVectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """数据摄入流水线.

    负责将 data_layer 的结构化数据 (股票/基金/宏观) 和外部文档
    统一导入检索系统。

    Usage::

        pipeline = IngestionPipeline(embedder, vector_store, text_store, config)
        pipeline.ingest_from_data_layer(db, config)
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        text_store: BaseTextStore,
        config: Optional[RetrievalConfig] = None,
    ):
        self.config = config or RetrievalConfig()
        self.indexer = Indexer(embedder, vector_store, text_store, self.config)

    # ================================================================
    # 从 data_layer 摄入
    # ================================================================

    def ingest_stocks(self, db) -> int:
        """导入股票基本信息到检索库.

        Args:
            db: DatabaseManager 实例.

        Returns:
            int: 索引的 chunk 数.
        """
        from data_layer.repositories.stock_repository import StockRepository

        stock_repo = StockRepository(db)
        stocks = stock_repo.find_all()

        if not stocks:
            logger.warning("No stocks found in database")
            return 0

        logger.info("Ingesting %d stock profiles...", len(stocks))
        return self.indexer.index_stock_profiles(stocks)

    def ingest_funds(self, db) -> int:
        """导入基金基本信息到检索库."""
        from data_layer.repositories.fund_repository import FundRepository

        fund_repo = FundRepository(db)
        funds = fund_repo.find_all()

        if not funds:
            logger.warning("No funds found in database")
            return 0

        # 构建基金文档
        from retrieval_layer.indexer.loader import Document

        documents = []
        for f in funds:
            parts = [f"基金代码: {f.code}"]
            if f.name:
                parts.append(f"名称: {f.name}")
            if f.fund_type:
                parts.append(f"类型: {f.fund_type}")
            if f.manager:
                parts.append(f"基金经理: {f.manager}")
            if f.company:
                parts.append(f"基金公司: {f.company}")

            documents.append(Document(
                content="；".join(parts),
                metadata={
                    "doc_type": "fund_profile",
                    "code": f.code,
                    "fund_type": f.fund_type or "",
                    "authority": 0.75,
                },
            ))

        logger.info("Ingesting %d fund profiles...", len(documents))
        return self.indexer.index_documents(documents)

    def ingest_from_directory(
        self,
        dir_path: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """导入外部文档目录 (研报 / 政策 / 新闻...).

        Args:
            dir_path: 文档目录路径.
            metadata: 全局元数据 (如 {'doc_type': 'research_report'}).

        Returns:
            int: 索引的 chunk 数.
        """
        return self.indexer.index_directory(
            dir_path, metadata=metadata,
        )

    def ingest_custom(
        self,
        records: list[dict],
        content_key: str = "content",
    ) -> int:
        """导入自定义数据 (如从 API 获取的金融新闻)."""
        return self.indexer.index_dicts(records, content_key)

    # ================================================================
    # 管理
    # ================================================================

    def clear_all(self) -> None:
        """清空所有检索数据."""
        self.indexer.vector_store.clear()
        self.indexer.text_store.clear()
        logger.info("All retrieval data cleared")

    def get_stats(self) -> dict:
        """获取检索系统统计."""
        return {
            "vector_store": self.indexer.vector_store.get_collection_stats(),
            "text_store_count": self.indexer.text_store.count(),
        }
