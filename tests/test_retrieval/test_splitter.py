# tests/test_retrieval/test_splitter.py
"""TextSplitter 单元测试."""

from retrieval_layer.indexer.loader import Document
from retrieval_layer.indexer.splitter import TextSplitter


class TestTextSplitter:
    """测试文本分块器."""

    def test_split_short_text(self):
        """短文本不应被分割."""
        splitter = TextSplitter(chunk_size=512, chunk_overlap=64)
        text = "这是一个短文本。"
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        """长文本应被分割为多个 chunk."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20, min_chunk_size=10)

        # 生成一段长文本
        text = "第一段落内容。" * 20 + "\n\n" + "第二段落内容。" * 20

        chunks = splitter.split_text(text)
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

        # 每个 chunk 不应该超过 chunk_size (允许一些误差)
        for c in chunks:
            assert len(c) <= 150, f"Chunk too long: {len(c)} chars"

    def test_split_documents_preserves_metadata(self):
        """分块后应保留原始 metadata."""
        splitter = TextSplitter(chunk_size=512, chunk_overlap=64)

        doc = Document(
            content="测试内容。",
            metadata={"source": "test", "authority": 0.9},
        )

        chunks = splitter.split_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test"
        assert chunks[0].metadata["authority"] == 0.9
        assert "chunk_index" in chunks[0].metadata

    def test_empty_document(self):
        """空文档应返回空列表."""
        splitter = TextSplitter(min_chunk_size=10)
        chunks = splitter.split_documents([Document(content="   ")])
        assert len(chunks) == 0

    def test_chinese_sentence_boundary(self):
        """中文文本应在句号处分割."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10, min_chunk_size=5)

        text = "第一句话的完整内容。第二句话也是完整内容。第三句话在这里。"

        chunks = splitter.split_text(text)
        # 各 chunk 应该尽量保持语义完整
        assert len(chunks) >= 1
        for c in chunks:
            assert len(c.strip()) > 0


class TestDocumentLoader:
    """测试文档加载器."""

    def test_load_from_dicts(self):
        """从字典列表加载文档."""
        from retrieval_layer.indexer.loader import DocumentLoader

        loader = DocumentLoader()
        records = [
            {"content": "文档1内容", "source": "test", "year": 2024},
            {"content": "文档2内容", "source": "test", "year": 2023},
        ]

        docs = loader.load_from_dicts(records, content_key="content")
        assert len(docs) == 2
        assert docs[0].content == "文档1内容"
        assert docs[0].metadata["source"] == "test"
        assert docs[0].metadata["year"] == 2024

    def test_load_stock_profiles(self):
        """从 Stock 对象构建文档."""
        from data_layer.models.stock import Stock
        from retrieval_layer.indexer.loader import DocumentLoader

        stocks = [
            Stock(symbol="600519", market="CN", name="贵州茅台",
                  exchange="SSE", sector="白酒", industry="食品饮料", currency="CNY"),
            Stock(symbol="AAPL", market="US", name="Apple Inc.",
                  exchange="NASDAQ", sector="Technology", currency="USD"),
        ]

        loader = DocumentLoader()
        docs = loader.load_stock_profiles(stocks)

        assert len(docs) == 2
        assert "600519" in docs[0].content
        assert "贵州茅台" in docs[0].content
        assert docs[0].metadata["doc_type"] == "stock_profile"
        assert docs[1].metadata["symbol"] == "AAPL"
