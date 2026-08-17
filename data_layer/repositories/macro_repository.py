# data_layer/repositories/macro_repository.py
"""宏观指标 & 财务指标仓库."""

import logging
from datetime import date
from typing import Optional

import pandas as pd

from data_layer.database.connection import DatabaseManager
from data_layer.models.macro import FinancialIndicator, MacroIndicator
from data_layer.models.price import _parse_date, _optional_float
from data_layer.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class MacroRepository(BaseRepository):
    """宏观指标 + 财务指标 CRUD.

    提供两个表的操作:
        - macro_indicators (宏观)
        - financial_indicators (上市公司财务)
    """

    table_name = "macro_indicators"
    columns = (
        "indicator_name", "indicator_value", "pub_date",
        "frequency", "source",
    )

    # ================================================================
    # 宏观指标
    # ================================================================

    def find_indicator(
        self, name: str, start_date: date, end_date: date
    ) -> list[MacroIndicator]:
        """查询某个宏观指标的历史数据."""
        rows = self._execute_all(
            "SELECT * FROM macro_indicators "
            "WHERE indicator_name = ? AND pub_date BETWEEN ? AND ? "
            "ORDER BY pub_date ASC",
            (name, start_date.isoformat(), end_date.isoformat()),
        )
        return [MacroIndicator.from_row(r) for r in rows]

    def find_indicator_as_df(
        self, name: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """查询宏观指标 (返回 DataFrame)."""
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM macro_indicators "
                "WHERE indicator_name = ? AND pub_date BETWEEN ? AND ? "
                "ORDER BY pub_date ASC",
                conn,
                params=(name, start_date.isoformat(), end_date.isoformat()),
            )
        if "pub_date" in df.columns:
            df["pub_date"] = pd.to_datetime(df["pub_date"])
        return df

    def get_latest_pub_date(self, indicator_name: str) -> Optional[date]:
        """获取某个指标的最新发布日期."""
        row = self._execute_scalar(
            "SELECT MAX(pub_date) FROM macro_indicators WHERE indicator_name = ?",
            (indicator_name,),
        )
        if row and row[0]:
            return _parse_date(row[0])
        return None

    def upsert_indicators(self, indicators: list[MacroIndicator]) -> int:
        """批量 upsert 宏观指标."""
        if not indicators:
            return 0

        records = []
        for ind in indicators:
            d = ind.to_dict()
            d["pub_date"] = ind.pub_date.isoformat() if ind.pub_date else None
            records.append(d)

        return self._insert_many(
            records, conflict_columns=("indicator_name", "pub_date"),
        )

    # ================================================================
    # 财务指标
    # ================================================================

    def find_financials(
        self, stock_id: int, report_type: Optional[str] = None
    ) -> list[FinancialIndicator]:
        """查询某只股票的财务指标."""
        if report_type:
            condition = "stock_id = ? AND report_type = ?"
            params = (stock_id, report_type)
        else:
            condition = "stock_id = ?"
            params = (stock_id,)

        rows = self._execute_all(
            f"SELECT * FROM financial_indicators WHERE {condition} "
            "ORDER BY report_date DESC",
            params,
        )
        return [FinancialIndicator.from_row(r) for r in rows]

    def find_latest_financials(
        self, stock_id: int
    ) -> Optional[FinancialIndicator]:
        """查询最新的财务指标."""
        row = self._execute_scalar(
            "SELECT * FROM financial_indicators WHERE stock_id = ? "
            "ORDER BY report_date DESC LIMIT 1",
            (stock_id,),
        )
        return FinancialIndicator.from_row(dict(row)) if row else None

    def upsert_financials(self, indicators: list[FinancialIndicator]) -> int:
        """批量 upsert 财务指标."""
        if not indicators:
            return 0

        records = []
        for ind in indicators:
            d = ind.to_dict()
            d["report_date"] = ind.report_date.isoformat() if ind.report_date else None
            records.append(d)

        # 在 financial_indicators 表上操作
        col_names = list(records[0].keys())
        placeholders = ", ".join(["?"] * len(col_names))
        cols = ", ".join(col_names)

        sql = (
            f"INSERT OR IGNORE INTO financial_indicators ({cols}) "
            f"VALUES ({placeholders})"
        )
        values = [tuple(r.get(c) for c in col_names) for r in records]

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, values)
            cursor.close()

        logger.debug("Upserted %d financial indicators", len(records))
        return len(records)
