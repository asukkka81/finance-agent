# data_layer/repositories/base.py
"""Repository 基类 — 提供通用 CRUD 模板方法."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from data_layer.database.connection import DatabaseManager

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """仓库基类.

    封装 SQLite 连接获取、批量插入、计数等通用操作。
    子类只需实现 table_name / columns，专注业务 SQL 即可。
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    # ----- 子类必须实现 -----

    @property
    @abstractmethod
    def table_name(self) -> str:
        """返回仓库对应的表名."""
        ...

    @property
    @abstractmethod
    def columns(self) -> tuple[str, ...]:
        """返回表的所有字段 (不含 id)."""
        ...

    # ----- 通用 CRUD -----

    def count(self) -> int:
        """查询表的总行数."""
        row = self._execute_scalar(f"SELECT COUNT(*) FROM {self.table_name}")
        return row[0] if row else 0

    def count_by_condition(self, condition: str, params: tuple = ()) -> int:
        """按条件计数."""
        row = self._execute_scalar(
            f"SELECT COUNT(*) FROM {self.table_name} WHERE {condition}",
            params,
        )
        return row[0] if row else 0

    def exists(self, condition: str, params: tuple = ()) -> bool:
        """检查是否存在符合条件的记录."""
        row = self._execute_scalar(
            f"SELECT 1 FROM {self.table_name} WHERE {condition} LIMIT 1",
            params,
        )
        return row is not None

    def truncate(self) -> None:
        """清空表数据 (保留表结构)."""
        with self.db.get_connection() as conn:
            conn.execute(f"DELETE FROM {self.table_name}")
        logger.info("Truncated table: %s", self.table_name)

    # ----- 内部辅助 -----

    def _execute_scalar(
        self, sql: str, params: tuple = ()
    ) -> Optional[Any]:
        """执行查询并返回第一条记录."""
        with self.db.get_connection() as conn:
            return conn.execute(sql, params).fetchone()

    def _execute_all(
        self, sql: str, params: tuple = ()
    ) -> list[dict]:
        """执行查询并返回所有记录为 dict 列表."""
        with self.db.get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    # ----- 批量 INSERT OR IGNORE -----

    def _insert_many(
        self, records: list[dict], conflict_columns: tuple[str, ...]
    ) -> int:
        """批量插入，遇到 conflict_columns 冲突时跳过 (INSERT OR IGNORE).

        Args:
            records: 待插入的记录列表.
            conflict_columns: 唯一约束的字段.

        Returns:
            int: 实际插入的行数.
        """
        if not records:
            return 0

        col_names = list(records[0].keys())
        placeholders = ", ".join(["?"] * len(col_names))
        cols = ", ".join(col_names)

        sql = f"INSERT OR IGNORE INTO {self.table_name} ({cols}) VALUES ({placeholders})"

        values = [tuple(r.get(c) for c in col_names) for r in records]

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # 在同一连接内计数，保证 WAL 模式下的可见性
            before_row = conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()
            before = before_row[0] if before_row else 0

            cursor.executemany(sql, values)

            after_row = conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()
            after = after_row[0] if after_row else 0

            inserted = after - before
            cursor.close()

        logger.debug(
            "Bulk insert into %s: %d submitted, %d inserted, %d skipped",
            self.table_name,
            len(records),
            inserted,
            len(records) - inserted,
        )
        return inserted
