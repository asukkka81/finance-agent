# tests/conftest.py
"""pytest 共享 fixtures."""

import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from data_layer.config import Config
from data_layer.database.connection import DatabaseManager
from data_layer.database.schema import CREATE_TABLE_STATEMENTS
from data_layer.repositories.fund_repository import FundRepository
from data_layer.repositories.macro_repository import MacroRepository
from data_layer.repositories.price_repository import PriceRepository
from data_layer.repositories.stock_repository import StockRepository


@pytest.fixture
def db_path():
    """临时 SQLite 数据库路径."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    yield temp_path
    # 清理
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def db(db_path):
    """已初始化 Schema 的 DatabaseManager."""
    manager = DatabaseManager(db_path)
    # 手动创建所有表
    with manager.get_connection() as conn:
        for ddl in CREATE_TABLE_STATEMENTS.values():
            conn.execute(ddl)
    return manager


@pytest.fixture
def stock_repo(db):
    """StockRepository 实例."""
    return StockRepository(db)


@pytest.fixture
def price_repo(db):
    """PriceRepository 实例."""
    return PriceRepository(db)


@pytest.fixture
def fund_repo(db):
    """FundRepository 实例."""
    return FundRepository(db)


@pytest.fixture
def macro_repo(db):
    """MacroRepository 实例."""
    return MacroRepository(db)


@pytest.fixture
def sample_config():
    """测试用 Config."""
    return Config({
        "database": {"path": ":memory:", "pragma": {}},
        "data_sources": {
            "yfinance": {"timeout": 10, "retry_count": 1, "retry_delay": 1},
            "akshare": {"timeout": 10, "retry_count": 1, "retry_delay": 1},
        },
        "sync": {
            "us_symbols": ["AAPL", "MSFT"],
            "cn_symbols": ["600519", "000858"],
            "fund_codes": ["000001"],
            "macro_indicators": ["cpi", "pmi"],
        },
        "logging": {"level": "DEBUG", "format": "%(message)s"},
    })


@pytest.fixture
def sample_stock_id(db):
    """在测试库中插入一只示例股票并返回 ID."""
    repo = StockRepository(db)
    from data_layer.models.stock import Stock
    return repo.upsert(Stock(symbol="AAPL", market="US", name="Apple Inc."))


@pytest.fixture
def today():
    return date.today()


@pytest.fixture
def last_week():
    return date.today() - timedelta(days=7)
