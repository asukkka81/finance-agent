# data_layer/repositories/stock_repository.py
"""股票元数据仓库."""

import logging
from typing import Optional

from data_layer.database.connection import DatabaseManager
from data_layer.models.stock import Stock
from data_layer.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class StockRepository(BaseRepository):
    """股票元数据 CRUD."""

    table_name = "stocks"
    columns = (
        "symbol", "name", "exchange", "market", "sector",
        "industry", "currency", "is_active",
    )

    # ----- 查询 -----

    def find_by_symbol(self, symbol: str) -> Optional[Stock]:
        """按股票代码查找."""
        row = self._execute_scalar(
            "SELECT * FROM stocks WHERE symbol = ?", (symbol,)
        )
        return Stock.from_row(dict(row)) if row else None

    def find_by_market(self, market: str) -> list[Stock]:
        """按市场查找所有股票."""
        rows = self._execute_all(
            "SELECT * FROM stocks WHERE market = ? ORDER BY symbol", (market,)
        )
        return [Stock.from_row(r) for r in rows]

    def find_all(self) -> list[Stock]:
        """查找所有股票."""
        rows = self._execute_all("SELECT * FROM stocks ORDER BY market, symbol")
        return [Stock.from_row(r) for r in rows]

    def search(self, keyword: str, limit: int = 20) -> list[Stock]:
        """模糊搜索 (名称或代码)."""
        pattern = f"%{keyword}%"
        rows = self._execute_all(
            "SELECT * FROM stocks WHERE symbol LIKE ? OR name LIKE ? "
            "ORDER BY symbol LIMIT ?",
            (pattern, pattern, limit),
        )
        return [Stock.from_row(r) for r in rows]

    # ----- 写入 -----

    def upsert(self, stock: Stock) -> int:
        """插入或更新股票元数据.

        如果 symbol 已存在则更新，否则插入。
        返回该股票的 id.
        """
        with self.db.get_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM stocks WHERE symbol = ?", (stock.symbol,)
            ).fetchone()

            if existing:
                stock_id = existing["id"]
                conn.execute(
                    """UPDATE stocks SET
                        name=?, exchange=?, sector=?, industry=?,
                        currency=?, is_active=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?""",
                    (
                        stock.name, stock.exchange, stock.sector,
                        stock.industry, stock.currency, int(stock.is_active),
                        stock_id,
                    ),
                )
                logger.debug("Updated stock: %s (id=%d)", stock.symbol, stock_id)
            else:
                data = stock.to_dict()
                cols = ", ".join(data.keys())
                placeholders = ", ".join(["?"] * len(data))
                cursor = conn.execute(
                    f"INSERT INTO stocks ({cols}) VALUES ({placeholders})",
                    tuple(data.values()),
                )
                stock_id = cursor.lastrowid
                logger.debug("Inserted stock: %s (id=%d)", stock.symbol, stock_id)

        return stock_id

    def upsert_batch(self, stocks: list[Stock]) -> list[int]:
        """批量 upsert 股票元数据. 返回所有 id 列表."""
        return [self.upsert(s) for s in stocks]

    def get_or_create(
        self, symbol: str, market: str, name: Optional[str] = None
    ) -> int:
        """获取 stock_id，如果不存在则自动创建.

        Args:
            symbol: 股票代码.
            market: 市场标识 'US' | 'CN'.
            name: 可选名称.

        Returns:
            int: stock_id.
        """
        stock = self.find_by_symbol(symbol)
        if stock:
            return stock.id

        # 自动推断 exchange
        exchange_map = {
            ("US",): "NASDAQ",
            ("CN", "6"): "SSE",   # 上交所
            ("CN", "0", "3"): "SZSE",  # 深交所
        }
        exchange = "UNKNOWN"
        if market == "US":
            exchange = "NASDAQ"
        elif market == "CN" and len(symbol) >= 1:
            prefix = symbol[0]
            if prefix == "6":
                exchange = "SSE"
            elif prefix in ("0", "3"):
                exchange = "SZSE"

        new_stock = Stock(
            symbol=symbol,
            market=market,
            name=name or symbol,
            exchange=exchange,
        )
        return self.upsert(new_stock)
