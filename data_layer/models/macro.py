# data_layer/models/macro.py
"""宏观指标 & 财务指标模型."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class MacroIndicator:
    """宏观经济指标.

    Attributes:
        indicator_name: 指标名称: CPI / PMI / M2 / LPR / GDP.
        pub_date: 发布日期.
        frequency: 频率: daily / weekly / monthly / quarterly / yearly.
    """

    indicator_name: str
    pub_date: date
    id: Optional[int] = None
    indicator_value: Optional[float] = None
    frequency: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: dict) -> "MacroIndicator":
        from data_layer.models.price import _parse_date, _optional_float

        return cls(
            id=row.get("id"),
            indicator_name=row["indicator_name"],
            indicator_value=_optional_float(row.get("indicator_value")),
            pub_date=_parse_date(row["pub_date"]),
            frequency=row.get("frequency"),
            source=row.get("source"),
            created_at=row.get("created_at"),
        )

    def to_dict(self) -> dict:
        return {
            "indicator_name": self.indicator_name,
            "indicator_value": self.indicator_value,
            "pub_date": self.pub_date.isoformat() if self.pub_date else None,
            "frequency": self.frequency,
            "source": self.source,
        }


@dataclass
class FinancialIndicator:
    """上市公司财务指标.

    Attributes:
        stock_id: 关联 stocks.id.
        report_date: 报告期 (如 2024-06-30).
        report_type: 年报 / 半年报 / 季报.
    """

    stock_id: int
    report_date: date
    report_type: str
    id: Optional[int] = None
    # 估值
    pe_ttm: Optional[float] = None
    pb: Optional[float] = None
    ps_ttm: Optional[float] = None
    # 盈利
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    # 成长
    revenue_yoy: Optional[float] = None
    profit_yoy: Optional[float] = None
    # 健康
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: dict) -> "FinancialIndicator":
        from data_layer.models.price import _parse_date, _optional_float

        return cls(
            id=row.get("id"),
            stock_id=row["stock_id"],
            report_date=_parse_date(row["report_date"]),
            report_type=row["report_type"],
            pe_ttm=_optional_float(row.get("pe_ttm")),
            pb=_optional_float(row.get("pb")),
            ps_ttm=_optional_float(row.get("ps_ttm")),
            roe=_optional_float(row.get("roe")),
            roa=_optional_float(row.get("roa")),
            gross_margin=_optional_float(row.get("gross_margin")),
            net_margin=_optional_float(row.get("net_margin")),
            revenue_yoy=_optional_float(row.get("revenue_yoy")),
            profit_yoy=_optional_float(row.get("profit_yoy")),
            debt_to_equity=_optional_float(row.get("debt_to_equity")),
            current_ratio=_optional_float(row.get("current_ratio")),
            created_at=row.get("created_at"),
        )

    def to_dict(self) -> dict:
        return {
            "stock_id": self.stock_id,
            "report_date": self.report_date.isoformat(),
            "report_type": self.report_type,
            "pe_ttm": self.pe_ttm,
            "pb": self.pb,
            "ps_ttm": self.ps_ttm,
            "roe": self.roe,
            "roa": self.roa,
            "gross_margin": self.gross_margin,
            "net_margin": self.net_margin,
            "revenue_yoy": self.revenue_yoy,
            "profit_yoy": self.profit_yoy,
            "debt_to_equity": self.debt_to_equity,
            "current_ratio": self.current_ratio,
        }
