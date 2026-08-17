# data_layer/models/fund.py
"""基金数据模型."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class Fund:
    """基金基础信息.

    Attributes:
        code: 基金代码，如 510300、000001.
        fund_type: 类型: 股票型/混合型/债券型/货币型/指数型/ETF 等.
    """

    code: str
    id: Optional[int] = None
    name: Optional[str] = None
    fund_type: Optional[str] = None
    manager: Optional[str] = None
    company: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: dict) -> "Fund":
        return cls(
            id=row.get("id"),
            code=row["code"],
            name=row.get("name"),
            fund_type=row.get("fund_type"),
            manager=row.get("manager"),
            company=row.get("company"),
            is_active=bool(row.get("is_active", 1)),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    def to_dict(self) -> dict:
        result = {"code": self.code}
        if self.name is not None:
            result["name"] = self.name
        if self.fund_type is not None:
            result["fund_type"] = self.fund_type
        if self.manager is not None:
            result["manager"] = self.manager
        if self.company is not None:
            result["company"] = self.company
        result["is_active"] = int(self.is_active)
        return result


@dataclass
class FundNAV:
    """基金净值数据.

    Attributes:
        fund_id: 关联的 funds.id.
        nav_date: 净值日期.
        unit_nav: 单位净值.
        accumulated_nav: 累计净值.
        daily_return: 日回报率 %.
    """

    fund_id: int
    nav_date: date
    id: Optional[int] = None
    unit_nav: Optional[float] = None
    accumulated_nav: Optional[float] = None
    daily_return: Optional[float] = None
    subscription: Optional[str] = None
    redemption: Optional[str] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: dict) -> "FundNAV":
        from data_layer.models.price import _parse_date, _optional_float

        return cls(
            id=row.get("id"),
            fund_id=row["fund_id"],
            nav_date=_parse_date(row["nav_date"]),
            unit_nav=_optional_float(row.get("unit_nav")),
            accumulated_nav=_optional_float(row.get("accumulated_nav")),
            daily_return=_optional_float(row.get("daily_return")),
            subscription=row.get("subscription"),
            redemption=row.get("redemption"),
            created_at=row.get("created_at"),
        )

    def to_dict(self) -> dict:
        return {
            "fund_id": self.fund_id,
            "nav_date": self.nav_date.isoformat() if self.nav_date else None,
            "unit_nav": self.unit_nav,
            "accumulated_nav": self.accumulated_nav,
            "daily_return": self.daily_return,
            "subscription": self.subscription,
            "redemption": self.redemption,
        }
