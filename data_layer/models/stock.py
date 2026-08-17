# data_layer/models/stock.py
"""股票元数据模型."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional


@dataclass
class Stock:
    """股票基础信息.

    Attributes:
        symbol: 股票代码，如 AAPL、600519.
        market: 市场: 'US' (美股) 或 'CN' (A股).
        exchange: 交易所，如 NASDAQ、SSE、SZSE.
    """

    symbol: str
    market: Literal["US", "CN"]
    id: Optional[int] = None
    name: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: str = "USD"
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: dict) -> "Stock":
        """从数据库行创建 Stock 实例."""
        return cls(
            id=row.get("id"),
            symbol=row["symbol"],
            name=row.get("name"),
            exchange=row.get("exchange"),
            market=row["market"],
            sector=row.get("sector"),
            industry=row.get("industry"),
            currency=row.get("currency", "USD"),
            is_active=bool(row.get("is_active", 1)),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    def to_dict(self) -> dict:
        """转为字典 (排除 None 值 & id)."""
        result = {
            "symbol": self.symbol,
            "market": self.market,
        }
        if self.name is not None:
            result["name"] = self.name
        if self.exchange is not None:
            result["exchange"] = self.exchange
        if self.sector is not None:
            result["sector"] = self.sector
        if self.industry is not None:
            result["industry"] = self.industry
        if self.currency != "USD":
            result["currency"] = self.currency
        result["is_active"] = int(self.is_active)
        return result
