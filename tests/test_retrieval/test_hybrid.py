# tests/test_retrieval/test_hybrid.py
"""混合检索 & RRF 融合测试."""

import pytest


class TestRRFFusion:
    """测试 RRF 融合算法."""

    def test_rrf_basic_fusion(self):
        """基本 RRF 融合."""
        from retrieval_layer.retrieval.hybrid import HybridRetriever

        # 模拟两路结果
        dense_results = [
            {"id": "d1", "document": "doc1", "score": 0.9},
            {"id": "d2", "document": "doc2", "score": 0.7},
            {"id": "d3", "document": "doc3", "score": 0.5},
        ]
        sparse_results = [
            {"id": "d2", "document": "doc2", "score": 0.8},
            {"id": "d4", "document": "doc4", "score": 0.6},
            {"id": "d1", "document": "doc1", "score": 0.3},
        ]

        # 创建一个 mock hybrid retriever
        hybrid = HybridRetriever.__new__(HybridRetriever)
        hybrid.rrf_k = 60

        fused = hybrid._rrf_fusion(dense_results, sparse_results)

        # d1 和 d2 都出现在两路，应该排前面
        assert len(fused) == 4  # d1, d2, d3, d4
        # d2 在两路都排第2 → score = 1/62 + 1/61
        # d1 排第1和第3 → score = 1/61 + 1/63
        top_ids = [f["id"] for f in fused[:2]]
        assert "d1" in top_ids
        assert "d2" in top_ids

    def test_rrf_match_type_marking(self):
        """融合结果应标记 match_type."""
        from retrieval_layer.retrieval.hybrid import HybridRetriever

        dense_results = [
            {"id": "both_doc", "document": "x", "score": 0.9},
            {"id": "dense_only", "document": "y", "score": 0.7},
        ]
        sparse_results = [
            {"id": "both_doc", "document": "x", "score": 0.8},
            {"id": "sparse_only", "document": "z", "score": 0.6},
        ]

        hybrid = HybridRetriever.__new__(HybridRetriever)
        hybrid.rrf_k = 60

        fused = hybrid._rrf_fusion(dense_results, sparse_results)

        match_types = {f["id"]: f.get("match_type") for f in fused}
        assert match_types["both_doc"] == "both"
        assert match_types["dense_only"] == "dense_only"
        assert match_types["sparse_only"] == "sparse_only"


class TestSearchPipeline:
    """测试 SearchPipeline."""

    def test_build_filter_single(self):
        """单个过滤条件."""
        from retrieval_layer.pipelines.search import SearchPipeline

        pipeline = SearchPipeline.__new__(SearchPipeline)
        where = pipeline._build_filter(doc_type="stock_analysis")
        assert where == {"doc_type": "stock_analysis"}

    def test_build_filter_multiple(self):
        """多个过滤条件 AND."""
        from retrieval_layer.pipelines.search import SearchPipeline

        pipeline = SearchPipeline.__new__(SearchPipeline)
        where = pipeline._build_filter(
            doc_type="stock_analysis", market="CN"
        )
        assert where == {"$and": [{"doc_type": "stock_analysis"}, {"market": "CN"}]}

    def test_build_filter_none(self):
        """无过滤."""
        from retrieval_layer.pipelines.search import SearchPipeline

        pipeline = SearchPipeline.__new__(SearchPipeline)
        where = pipeline._build_filter()
        assert where is None

    def test_post_filter_authority(self):
        """权威度过滤."""
        from retrieval_layer.pipelines.search import SearchPipeline

        pipeline = SearchPipeline.__new__(SearchPipeline)
        candidates = [
            {"document": "x", "score": 0.9, "metadata": {"authority": 0.9}},
            {"document": "y", "score": 0.7, "metadata": {"authority": 0.3}},
            {"document": "z", "score": 0.5, "metadata": {}},  # default 0.5
        ]

        filtered = pipeline._post_filter(candidates, min_authority=0.5)
        assert len(filtered) == 2
        ids = {c["document"] for c in filtered}
        assert ids == {"x", "z"}  # y filtered out (0.3 < 0.5)
