# tests/test_services/test_data_service.py
"""DataService 核心服务测试."""

import pytest

from data_layer.models.stock import Stock
from data_layer.services.data_service import DataService


class TestDataService:
    """测试 DataService 的核心功能."""

    @pytest.fixture
    def service(self, db, sample_config):
        """创建 DataService 实例."""
        return DataService(db, sample_config)

    def test_search_stocks(self, service, stock_repo):
        """搜索股票应返回匹配结果."""
        stock_repo.upsert(Stock(symbol="AAPL", market="US", name="Apple Inc."))
        stock_repo.upsert(Stock(symbol="MSFT", market="US", name="Microsoft Corp."))
        stock_repo.upsert(Stock(symbol="600519", market="CN", name="贵州茅台"))

        # 按名称搜索
        results = service.search_stocks("Apple")
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"

        # 按代码搜
        results = service.search_stocks("6005")
        assert len(results) == 1
        assert results[0]["name"] == "贵州茅台"

        # 空结果
        results = service.search_stocks("ZZZZZ")
        assert len(results) == 0

    def test_get_price_history_empty(self, service):
        """无数据时应返回空 DataFrame."""
        import pandas as pd
        from datetime import date

        df = service.get_price_history(
            "NONEXISTENT", date(2024, 1, 1), date(2024, 12, 31)
        )
        assert df.empty

    def test_get_latest_prices_empty(self, service):
        """无数据时应返回空 DataFrame."""
        df = service.get_latest_prices("US")
        assert df.empty
