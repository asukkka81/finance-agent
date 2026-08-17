# tests/test_retrieval/conftest.py
"""检索层测试 fixtures."""

import tempfile
from pathlib import Path

import pytest

from retrieval_layer.config import RetrievalConfig
from retrieval_layer.stores.chroma_store import ChromaVectorStore
from retrieval_layer.stores.fts5_store import FTS5Index


@pytest.fixture
def retrieval_config():
    """测试用 RetrievalConfig (临时目录)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield RetrievalConfig(
            chroma_path=f"{tmpdir}/chroma_test",
            fts5_db_path=f"{tmpdir}/fts5_test.db",
        )


@pytest.fixture
def sample_documents():
    """测试用金融文档."""
    return [
        {
            "id": "doc_001",
            "content": "贵州茅台（600519）是中国白酒行业的龙头企业，"
                       "2024年实现营业总收入1741亿元，同比增长15.7%。"
                       "公司ROE维持在34%以上，PE_TTM约为28倍。",
            "metadata": {
                "doc_type": "stock_analysis",
                "symbol": "600519",
                "market": "CN",
                "authority": 0.9,
            },
        },
        {
            "id": "doc_002",
            "content": "苹果公司（AAPL）是全球市值最大的科技公司，"
                       "2024财年营收3910亿美元，iPhone收入占比约52%。"
                       "公司在AI领域的布局逐步深化。",
            "metadata": {
                "doc_type": "stock_analysis",
                "symbol": "AAPL",
                "market": "US",
                "authority": 0.85,
            },
        },
        {
            "id": "doc_003",
            "content": "市盈率（PE）是股票估值最常用的指标之一，"
                       "计算公式为：PE = 股价 / 每股收益。"
                       "PE越低通常表示估值越低，但需要结合行业和成长性综合判断。",
            "metadata": {
                "doc_type": "financial_knowledge",
                "authority": 0.95,
            },
        },
        {
            "id": "doc_004",
            "content": "沪深300指数由沪深两市中市值最大、流动性最好的300只股票组成，"
                       "覆盖A股市场约60%的总市值，是衡量A股整体表现的重要基准。",
            "metadata": {
                "doc_type": "financial_knowledge",
                "market": "CN",
                "authority": 0.9,
            },
        },
        {
            "id": "doc_005",
            "content": "ROE（净资产收益率）是衡量公司盈利能力的关键指标，"
                       "计算公式为：ROE = 净利润 / 净资产 × 100%。"
                       "巴菲特偏好ROE持续高于15%的公司。"
                       "贵州茅台的ROE长期维持在30%以上，属于A股顶尖水平。",
            "metadata": {
                "doc_type": "financial_knowledge",
                "authority": 0.95,
            },
        },
    ]
