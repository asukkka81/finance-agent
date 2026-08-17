# tests/test_database/test_connection.py
"""DatabaseManager 单元测试."""

import pytest

from data_layer.database.connection import DatabaseManager
from data_layer.database.schema import ALL_TABLES


class TestDatabaseManager:
    """测试数据库连接 & Schema 初始化."""

    def test_initialize_schema_creates_all_tables(self, db):
        """初始化后所有表应被创建."""
        tables = db.get_all_tables()
        for table_name in ALL_TABLES:
            assert table_name in tables, f"Table '{table_name}' should exist"

    def test_get_connection_context_manager(self, db):
        """上下文管理器应自动提交."""
        with db.get_connection() as conn:
            conn.execute(
                "INSERT INTO stocks (symbol, market) VALUES (?, ?)",
                ("TEST", "US"),
            )
        # 退出上下文后数据应已持久化
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM stocks WHERE symbol=?", ("TEST",)
            ).fetchone()
            assert row is not None
            assert row["symbol"] == "TEST"

    def test_get_connection_rollback_on_error(self, db):
        """发生异常时应自动回滚."""
        with pytest.raises(Exception):
            with db.get_connection() as conn:
                conn.execute(
                    "INSERT INTO stocks (symbol, market) VALUES (?, ?)",
                    ("TEST2", "US"),
                )
                raise ValueError("rollback test")

        # 数据不应存在
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM stocks WHERE symbol=?", ("TEST2",)
            ).fetchone()
            assert row is None

    def test_table_exists(self, db):
        """table_exists 应正确判断."""
        assert db.table_exists("stocks") is True
        assert db.table_exists("nonexistent_table") is False

    def test_get_table_count(self, db):
        """get_table_count 应返回正确行数."""
        assert db.get_table_count("stocks") == 0
        with db.get_connection() as conn:
            conn.execute("INSERT INTO stocks (symbol, market) VALUES ('AAPL', 'US')")
        assert db.get_table_count("stocks") == 1

    def test_unique_constraint(self, db):
        """UNIQUE 索引应防止重复 symbol."""
        with db.get_connection() as conn:
            conn.execute("INSERT INTO stocks (symbol, market) VALUES ('AAPL', 'US')")
            with pytest.raises(Exception):
                conn.execute(
                    "INSERT INTO stocks (symbol, market) VALUES ('AAPL', 'US')"
                )
