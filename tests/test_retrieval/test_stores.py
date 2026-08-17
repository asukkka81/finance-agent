# tests/test_retrieval/test_stores.py
"""存储层单元测试 (FTS5 + ChromaDB)."""

import numpy as np
import pytest


class TestFTS5Index:
    """测试 FTS5 全文索引."""

    def test_add_and_count(self, retrieval_config):
        """添加文档后 count 应正确."""
        from retrieval_layer.stores.fts5_store import FTS5Index

        fts = FTS5Index(retrieval_config.fts5_db_path)
        fts.clear()

        fts.add(
            ids=["doc1", "doc2"],
            documents=["贵州茅台是中国白酒龙头企业", "苹果公司市值全球第一"],
        )
        assert fts.count() == 2

    def test_search_chinese(self, retrieval_config):
        """中文检索应返回匹配结果."""
        from retrieval_layer.stores.fts5_store import FTS5Index

        fts = FTS5Index(retrieval_config.fts5_db_path)
        fts.clear()

        fts.add(
            ids=["doc1", "doc2", "doc3"],
            documents=[
                "贵州茅台2024年营收1741亿元",
                "五粮液是中国第二大白酒品牌",
                "苹果公司发布最新iPhone",
            ],
        )

        results = fts.search("茅台 营收", top_k=5)
        assert len(results) > 0
        assert "doc1" in [r["id"] for r in results]

    def test_search_english(self, retrieval_config):
        """英文检索应返回匹配结果."""
        from retrieval_layer.stores.fts5_store import FTS5Index

        fts = FTS5Index(retrieval_config.fts5_db_path)
        fts.clear()

        fts.add(
            ids=["doc1", "doc2"],
            documents=[
                "Apple Inc. reported record revenue in Q4 2024",
                "Microsoft Azure cloud growth accelerates",
            ],
        )

        results = fts.search("Apple revenue", top_k=5)
        assert len(results) > 0
        assert results[0]["id"] == "doc1"

    def test_delete(self, retrieval_config):
        """删除后 count 应减少."""
        from retrieval_layer.stores.fts5_store import FTS5Index

        fts = FTS5Index(retrieval_config.fts5_db_path)
        fts.clear()

        fts.add(ids=["doc1", "doc2"], documents=["内容1", "内容2"])
        assert fts.count() == 2

        fts.delete(["doc1"])
        assert fts.count() == 1

    def test_phrase_search(self, retrieval_config):
        """精确短语搜索."""
        from retrieval_layer.stores.fts5_store import FTS5Index

        fts = FTS5Index(retrieval_config.fts5_db_path)
        fts.clear()

        fts.add(
            ids=["doc1"],
            documents=["贵州茅台是中国白酒行业的龙头企业"],
        )

        results = fts.phrase_search("贵州茅台", top_k=5)
        assert len(results) > 0


class TestChromaVectorStore:
    """测试 ChromaDB 向量存储."""

    def test_add_and_count(self, retrieval_config):
        """添加向量后 count 应正确."""
        from retrieval_layer.stores.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            path=retrieval_config.chroma_path,
            collection_name="test_collection",
        )
        store.clear()

        vectors = np.random.randn(3, 128).astype(np.float32)
        store.add(
            ids=["v1", "v2", "v3"],
            vectors=vectors,
            documents=["文档1", "文档2", "文档3"],
        )
        assert store.count() == 3

    def test_search_returns_results(self, retrieval_config):
        """向量检索应返回相似结果."""
        from retrieval_layer.stores.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            path=retrieval_config.chroma_path,
            collection_name="test_search",
        )
        store.clear()

        # 添加已知向量
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        store.add(
            ids=["v1", "v2", "v3"],
            vectors=vectors,
            documents=["文档A", "文档B", "文档C"],
        )

        # 查询接近 v1 的向量
        query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        results = store.search(query, top_k=2)

        assert len(results) == 2
        # v1 应该排第一
        assert results[0]["id"] == "v1"
        assert results[0]["score"] > results[1]["score"]

    def test_search_with_metadata_filter(self, retrieval_config):
        """元数据过滤."""
        from retrieval_layer.stores.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            path=retrieval_config.chroma_path,
            collection_name="test_filter",
        )
        store.clear()

        vectors = np.random.randn(4, 128).astype(np.float32)
        store.add(
            ids=["a1", "a2", "b1", "b2"],
            vectors=vectors,
            documents=["A类1", "A类2", "B类1", "B类2"],
            metadatas=[
                {"category": "A"}, {"category": "A"},
                {"category": "B"}, {"category": "B"},
            ],
        )

        query = np.random.randn(128).astype(np.float32)
        results = store.search(query, top_k=10, where={"category": "A"})

        assert len(results) > 0
        for r in results:
            assert r["metadata"]["category"] == "A"
