# data_layer/database/connection.py
"""SQLite 连接管理器 — 线程安全，支持上下文管理 & Schema 初始化."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from data_layer.database.schema import CREATE_TABLE_STATEMENTS, ALL_TABLES

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite 数据库连接管理器.

    功能:
        - 提供线程安全的连接上下文
        - 自动启用 WAL 模式 & 外键约束
        - 首次运行时自动创建所有表

    Usage::

        db = DatabaseManager("data/finance.db")
        db.initialize_schema()

        with db.get_connection() as conn:
            conn.execute("SELECT * FROM stocks")
    """

    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).resolve())
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        """确保数据库文件所在目录存在."""
        parent = Path(self.db_path).parent
        parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> sqlite3.Connection:
        """获取数据库连接上下文.

        Yields:
            sqlite3.Connection: 已配置 WAL 模式 & 外键约束的连接.

        上下文退出时自动 commit 或 rollback.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")  # 64MB
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """初始化数据库 Schema — 创建所有表 & 索引 (IF NOT EXISTS)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for table_name, ddl in CREATE_TABLE_STATEMENTS.items():
                cursor.execute(ddl)
            cursor.close()
        logger.info(
            "Schema initialized: %d tables created at %s",
            len(CREATE_TABLE_STATEMENTS),
            self.db_path,
        )

    def table_exists(self, table_name: str) -> bool:
        """检查指定表是否存在."""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            return result is not None

    def get_table_count(self, table_name: str) -> int:
        """获取某张表的行数."""
        with self.get_connection() as conn:
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return result[0] if result else 0

    def get_all_tables(self) -> list[str]:
        """获取数据库中所有用户表的名称."""
        with self.get_connection() as conn:
            results = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            return [r["name"] for r in results]
