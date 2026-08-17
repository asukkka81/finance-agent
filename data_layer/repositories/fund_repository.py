# data_layer/repositories/fund_repository.py
"""基金数据仓库."""

import logging
from datetime import date
from typing import Optional

import pandas as pd

from data_layer.database.connection import DatabaseManager
from data_layer.models.fund import Fund, FundNAV
from data_layer.models.price import _parse_date, _optional_float
from data_layer.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class FundRepository(BaseRepository):
    """基金元数据 & 净值 CRUD."""

    table_name = "funds"
    columns = ("code", "name", "fund_type", "manager", "company", "is_active")

    # ===== 基金元数据 =====

    def find_by_code(self, code: str) -> Optional[Fund]:
        row = self._execute_scalar(
            "SELECT * FROM funds WHERE code = ?", (code,)
        )
        return Fund.from_row(dict(row)) if row else None

    def find_all(self) -> list[Fund]:
        rows = self._execute_all("SELECT * FROM funds ORDER BY code")
        return [Fund.from_row(r) for r in rows]

    def upsert(self, fund: Fund) -> int:
        """插入或更新基金元数据，返回 fund_id."""
        with self.db.get_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM funds WHERE code = ?", (fund.code,)
            ).fetchone()

            if existing:
                fund_id = existing["id"]
                conn.execute(
                    """UPDATE funds SET name=?, fund_type=?, manager=?,
                       company=?, is_active=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?""",
                    (fund.name, fund.fund_type, fund.manager,
                     fund.company, int(fund.is_active), fund_id),
                )
            else:
                data = fund.to_dict()
                cols = ", ".join(data.keys())
                placeholders = ", ".join(["?"] * len(data))
                cursor = conn.execute(
                    f"INSERT INTO funds ({cols}) VALUES ({placeholders})",
                    tuple(data.values()),
                )
                fund_id = cursor.lastrowid
        return fund_id

    def get_or_create(self, code: str, name: Optional[str] = None) -> int:
        fund = self.find_by_code(code)
        if fund:
            return fund.id
        return self.upsert(Fund(code=code, name=name or code))

    # ===== 基金净值 =====

    def find_nav_history(
        self, fund_id: int, start_date: date, end_date: date
    ) -> list[FundNAV]:
        rows = self._execute_all(
            "SELECT * FROM fund_nav WHERE fund_id = ? AND nav_date BETWEEN ? AND ? "
            "ORDER BY nav_date ASC",
            (fund_id, start_date.isoformat(), end_date.isoformat()),
        )
        return [FundNAV.from_row(r) for r in rows]

    def find_nav_history_as_df(
        self, fund_id: int, start_date: date, end_date: date
    ) -> pd.DataFrame:
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM fund_nav WHERE fund_id = ? AND nav_date BETWEEN ? AND ? "
                "ORDER BY nav_date ASC",
                conn,
                params=(fund_id, start_date.isoformat(), end_date.isoformat()),
            )
        if "nav_date" in df.columns:
            df["nav_date"] = pd.to_datetime(df["nav_date"])
        return df

    def get_latest_nav_date(self, fund_id: int) -> Optional[date]:
        row = self._execute_scalar(
            "SELECT MAX(nav_date) FROM fund_nav WHERE fund_id = ?", (fund_id,)
        )
        if row and row[0]:
            return _parse_date(row[0])
        return None

    def upsert_nav_batch(self, nav_list: list[FundNAV]) -> int:
        """批量 upsert 基金净值.

        注意: 直接操作 fund_nav 表，不走基类的 _insert_many，
        因为基类的 self.table_name 指向 'funds' 表.
        """
        if not nav_list:
            return 0

        col_names = [
            "fund_id", "nav_date", "unit_nav", "accumulated_nav",
            "daily_return", "subscription", "redemption",
        ]
        placeholders = ", ".join(["?"] * len(col_names))
        cols = ", ".join(col_names)

        values = []
        for nav in nav_list:
            row_dict = nav.to_dict()
            row_dict["nav_date"] = (
                nav.nav_date.isoformat() if nav.nav_date else None
            )
            values.append(tuple(row_dict.get(c) for c in col_names))

        sql = f"INSERT OR IGNORE INTO fund_nav ({cols}) VALUES ({placeholders})"

        # 手动计算插入数
        before = self._count_table("fund_nav")
        with self.db.get_connection() as conn:
            conn.executemany(sql, values)
        after = self._count_table("fund_nav")

        inserted = after - before
        logger.debug(
            "Bulk insert fund_nav: %d submitted, %d inserted, %d skipped",
            len(values), inserted, len(values) - inserted,
        )
        return inserted

    def _count_table(self, tbl: str) -> int:
        """内部辅助: 查指定表的行数."""
        row = self._execute_scalar(f"SELECT COUNT(*) FROM {tbl}")
        return row[0] if row else 0
