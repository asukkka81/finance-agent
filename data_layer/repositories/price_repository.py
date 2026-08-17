# data_layer/repositories/price_repository.py
"""日线行情数据仓库."""

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd

from data_layer.database.connection import DatabaseManager
from data_layer.models.price import DailyPrice, _parse_date, _optional_float
from data_layer.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class PriceRepository(BaseRepository):
    """日线行情 CRUD.

    去重策略: (stock_id, trade_date) UNIQUE 索引 + INSERT OR IGNORE.
    """

    table_name = "daily_prices"
    columns = (
        "stock_id", "trade_date", "open", "high", "low", "close",
        "volume", "adj_close", "pre_close", "change_pct", "turnover_rate",
    )
    _conflict_cols = ("stock_id", "trade_date")

    # ----- 查询 -----

    def find_by_stock_and_date(
        self, stock_id: int, trade_date: date
    ) -> Optional[DailyPrice]:
        """根据 stock_id + 日期查单条记录."""
        row = self._execute_scalar(
            "SELECT * FROM daily_prices WHERE stock_id = ? AND trade_date = ?",
            (stock_id, trade_date.isoformat()),
        )
        return DailyPrice.from_row(dict(row)) if row else None

    def find_history(
        self,
        stock_id: int,
        start_date: date,
        end_date: date,
    ) -> list[DailyPrice]:
        """查询日期范围内的行情数据."""
        rows = self._execute_all(
            "SELECT * FROM daily_prices "
            "WHERE stock_id = ? AND trade_date BETWEEN ? AND ? "
            "ORDER BY trade_date ASC",
            (stock_id, start_date.isoformat(), end_date.isoformat()),
        )
        return [DailyPrice.from_row(r) for r in rows]

    def find_history_as_df(
        self,
        stock_id: int,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """查询日期范围内的行情数据 (返回 DataFrame)."""
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM daily_prices "
                "WHERE stock_id = ? AND trade_date BETWEEN ? AND ? "
                "ORDER BY trade_date ASC",
                conn,
                params=(stock_id, start_date.isoformat(), end_date.isoformat()),
            )
        # 日期列转为 datetime
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def find_latest_batch(
        self, stock_ids: list[int], days: int = 1
    ) -> pd.DataFrame:
        """批量获取多只股票的最近 N 天行情."""
        if not stock_ids:
            return pd.DataFrame()

        placeholders = ", ".join(["?"] * len(stock_ids))
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(
                f"""SELECT dp.*, s.symbol, s.name
                FROM daily_prices dp
                JOIN stocks s ON dp.stock_id = s.id
                WHERE dp.stock_id IN ({placeholders})
                AND dp.trade_date >= date('now', '-{days * 3} days')
                ORDER BY dp.stock_id, dp.trade_date DESC
                """,
                conn,
                params=stock_ids,
            )
        return df

    def get_latest_trade_date(self, stock_id: int) -> Optional[date]:
        """获取某只股票的最新数据日期 (用于增量更新)."""
        row = self._execute_scalar(
            "SELECT MAX(trade_date) FROM daily_prices WHERE stock_id = ?",
            (stock_id,),
        )
        if row and row[0]:
            return _parse_date(row[0])
        return None

    def get_latest_trade_date_by_market(self, market: str) -> Optional[date]:
        """获取某市场最新数据的公共日期.

        返回该市场所有股票 latest_date 的最小值 (即可安全增量拉取的最早断点).
        """
        row = self._execute_scalar(
            """SELECT MIN(latest) FROM (
                SELECT MAX(dp.trade_date) as latest
                FROM daily_prices dp
                JOIN stocks s ON dp.stock_id = s.id
                WHERE s.market = ?
                GROUP BY dp.stock_id
            )""",
            (market,),
        )
        if row and row[0]:
            return _parse_date(row[0])
        return None

    # ----- 写入 -----

    def upsert_prices(self, prices: list[DailyPrice]) -> int:
        """批量 upsert 行情数据. 利用 UNIQUE 索引自动跳过重复.

        Returns:
            int: 实际插入的记录数.
        """
        if not prices:
            return 0

        records = []
        for p in prices:
            d = p.to_dict()
            d["trade_date"] = (
                p.trade_date.isoformat() if p.trade_date else None
            )
            records.append(d)

        return self._insert_many(records, self._conflict_cols)

    def upsert_prices_from_df(
        self, df: pd.DataFrame, stock_id_map: dict[str, int]
    ) -> dict[str, int]:
        """从 DataFrame 批量 upsert 行情数据.

        这是从 Fetcher → Database 的主要写入路径.

        Args:
            df: 标准化的行情 DataFrame，包含 symbol, trade_date, OHLCV 等列.
            stock_id_map: {symbol: stock_id} 映射.

        Returns:
            dict: {'inserted': N, 'skipped': M} 统计信息.
        """
        if df.empty:
            return {"inserted": 0, "skipped": 0}

        records = []
        for _, row in df.iterrows():
            symbol = row.get("symbol")
            sid = stock_id_map.get(symbol)
            if sid is None:
                continue

            records.append({
                "stock_id": sid,
                "trade_date": (
                    _parse_date(row["trade_date"]).isoformat()
                    if row.get("trade_date") else None
                ),
                "open": _optional_float(row.get("open")),
                "high": _optional_float(row.get("high")),
                "low": _optional_float(row.get("low")),
                "close": _optional_float(row.get("close")),
                "volume": _optional_float(row.get("volume")),
                "adj_close": _optional_float(row.get("adj_close")),
                "pre_close": _optional_float(row.get("pre_close")),
                "change_pct": _optional_float(row.get("change_pct")),
                "turnover_rate": _optional_float(row.get("turnover_rate")),
            })

        total = len(records)
        inserted = self._insert_many(records, self._conflict_cols)
        return {"inserted": inserted, "skipped": total - inserted}

    # ----- 统计 -----

    def get_date_range(self, stock_id: int) -> tuple[Optional[date], Optional[date]]:
        """获取某只股票的数据日期范围."""
        row = self._execute_scalar(
            "SELECT MIN(trade_date), MAX(trade_date) FROM daily_prices WHERE stock_id = ?",
            (stock_id,),
        )
        if row and row[0]:
            return _parse_date(row[0]), _parse_date(row[1])
        return None, None

    def get_record_count(self, stock_id: int) -> int:
        """获取某只股票的行情记录数."""
        return self.count_by_condition("stock_id = ?", (stock_id,))
