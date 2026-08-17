# tests/test_repositories/test_price_repository.py
"""PriceRepository 单元测试."""

from datetime import date, timedelta

import pytest

from data_layer.models.price import DailyPrice
from data_layer.models.stock import Stock


class TestPriceRepository:
    """测试行情仓库的 CRUD 操作."""

    @pytest.fixture
    def stock_id(self, stock_repo):
        """创建测试股票并返回 ID."""
        return stock_repo.upsert(
            Stock(symbol="AAPL", market="US", name="Apple Inc.")
        )

    @pytest.fixture
    def sample_prices(self, stock_id):
        """生成 5 条测试行情数据."""
        base = date(2024, 1, 2)
        prices = []
        for i in range(5):
            d = base + timedelta(days=i)
            prices.append(DailyPrice(
                stock_id=stock_id,
                trade_date=d,
                open=100.0 + i,
                high=105.0 + i,
                low=98.0 + i,
                close=103.0 + i,
                volume=1000000 + i * 10000,
            ))
        return prices

    def test_upsert_prices_inserts_new_records(self, price_repo, sample_prices):
        """新增行情记录."""
        inserted = price_repo.upsert_prices(sample_prices)
        assert inserted == 5

    def test_upsert_prices_skips_duplicates(self, price_repo, sample_prices):
        """重复插入应被 UNIQUE 索引跳过."""
        # 第一次插入
        price_repo.upsert_prices(sample_prices)
        # 第二次插入相同数据
        inserted = price_repo.upsert_prices(sample_prices)
        assert inserted == 0  # 全跳过

    def test_find_history_returns_correct_range(
        self, price_repo, sample_prices, stock_id
    ):
        """查询日期范围应返回正确的数据."""
        price_repo.upsert_prices(sample_prices)

        result = price_repo.find_history(
            stock_id,
            date(2024, 1, 2),
            date(2024, 1, 4),
        )
        assert len(result) == 3
        assert result[0].trade_date == date(2024, 1, 2)
        assert result[-1].trade_date == date(2024, 1, 4)

    def test_get_latest_trade_date(self, price_repo, sample_prices, stock_id):
        """应返回最新交易日期."""
        price_repo.upsert_prices(sample_prices)
        latest = price_repo.get_latest_trade_date(stock_id)
        assert latest == date(2024, 1, 6)

    def test_get_latest_trade_date_empty(self, price_repo, stock_id):
        """无数据时应返回 None."""
        latest = price_repo.get_latest_trade_date(stock_id)
        assert latest is None

    def test_get_date_range(self, price_repo, sample_prices, stock_id):
        """应返回完整的日期范围."""
        price_repo.upsert_prices(sample_prices)
        start, end = price_repo.get_date_range(stock_id)
        assert start == date(2024, 1, 2)
        assert end == date(2024, 1, 6)

    def test_count(self, price_repo, sample_prices):
        """计数应准确."""
        price_repo.upsert_prices(sample_prices)
        assert price_repo.count() == 5
