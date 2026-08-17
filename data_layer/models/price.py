# data_layer/models/price.py
"""日线行情数据模型."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


@dataclass
class DailyPrice:
    """股票日线行情 (OHLCV + 附加指标).

    Attributes:
        stock_id: 关联的 stocks.id.
        trade_date: 交易日期.
        adj_close: 复权收盘价 (yfinance).
        pre_close: 前收盘价 (akshare).
        change_pct: 涨跌幅 %.
        turnover_rate: 换手率 %.
    """

    stock_id: int
    trade_date: date
    id: Optional[int] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    adj_close: Optional[float] = None
    pre_close: Optional[float] = None
    change_pct: Optional[float] = None
    turnover_rate: Optional[float] = None
    created_at: Optional[datetime] = None

    # ----- 工厂方法 -----

    @classmethod
    def from_row(cls, row: dict) -> "DailyPrice":
        """从数据库行创建 DailyPrice 实例."""
        return cls(
            id=row.get("id"),
            stock_id=row["stock_id"],
            trade_date=_parse_date(row["trade_date"]),
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=row.get("volume"),
            adj_close=row.get("adj_close"),
            pre_close=row.get("pre_close"),
            change_pct=row.get("change_pct"),
            turnover_rate=row.get("turnover_rate"),
            created_at=row.get("created_at"),
        )

    @classmethod
    def from_dataframe_row(cls, row: dict, stock_id: int) -> "DailyPrice":
        """从 DataFrame 行 + stock_id 创建实例."""
        return cls(
            stock_id=stock_id,
            trade_date=_parse_date(row.get("trade_date")),
            open=_optional_float(row.get("open")),
            high=_optional_float(row.get("high")),
            low=_optional_float(row.get("low")),
            close=_optional_float(row.get("close")),
            volume=_optional_float(row.get("volume")),
            adj_close=_optional_float(row.get("adj_close")),
            pre_close=_optional_float(row.get("pre_close")),
            change_pct=_optional_float(row.get("change_pct")),
            turnover_rate=_optional_float(row.get("turnover_rate")),
        )

    # ----- 序列化 -----

    def to_dict(self) -> dict:
        """转为字典 (不含 id & created_at)."""
        return {
            "stock_id": self.stock_id,
            "trade_date": (
                self.trade_date.isoformat() if self.trade_date else None
            ),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adj_close": self.adj_close,
            "pre_close": self.pre_close,
            "change_pct": self.change_pct,
            "turnover_rate": self.turnover_rate,
        }


# ----- 内部工具函数 -----

def _parse_date(value) -> date:
    """将各种日期格式统一转为 date 对象."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        # 尝试多种格式
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        # 最后尝试 pandas Timestamp
        import pandas as pd
        return pd.Timestamp(value).date()
    import pandas as pd
    return pd.Timestamp(value).date()


def _optional_float(value) -> Optional[float]:
    """安全转为 float，处理 NaN/None."""
    if value is None:
        return None
    try:
        result = float(value)
        import math
        return None if math.isnan(result) else result
    except (ValueError, TypeError):
        return None
