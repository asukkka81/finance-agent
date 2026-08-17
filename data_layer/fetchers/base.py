# data_layer/fetchers/base.py
"""数据获取器抽象基类 — 统一数据源接口."""

import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd

from data_layer.models.stock import Stock

logger = logging.getLogger(__name__)


class AbstractFetcher(ABC):
    """金融数据获取器抽象基类.

    每种数据源 (yfinance, akshare, ...) 实现此接口。
    所有方法返回标准化的 pd.DataFrame 或 Stock 列表。
    """

    # ----- 元信息 -----

    @property
    @abstractmethod
    def market(self) -> str:
        """返回市场标识: 'US' | 'CN'."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """返回数据源名称: 'yfinance' | 'akshare'."""
        ...

    # ----- 数据获取 -----

    @abstractmethod
    def fetch_daily_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取股票日线行情.

        Args:
            symbols: 股票代码列表.
            start_date: 开始日期.
            end_date: 结束日期.

        Returns:
            标准化的 DataFrame，必须包含列:
                symbol, trade_date, open, high, low, close, volume,
                [adj_close, pre_close, change_pct, turnover_rate]
        """
        ...

    @abstractmethod
    def fetch_stock_info(self, symbols: list[str]) -> list[Stock]:
        """获取股票基础信息.

        Args:
            symbols: 股票代码列表.

        Returns:
            Stock 对象列表.
        """
        ...

    # ----- 可选接口 -----

    def fetch_fund_nav(
        self,
        fund_codes: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取基金净值 (仅 akshare 支持)."""
        raise NotImplementedError(
            f"{self.source_name} does not support fund NAV fetching"
        )

    def fetch_macro_indicator(
        self, indicator_name: str
    ) -> pd.DataFrame:
        """获取宏观指标 (仅 akshare 支持)."""
        raise NotImplementedError(
            f"{self.source_name} does not support macro indicator fetching"
        )

    def fetch_financial_indicators(
        self, symbols: list[str]
    ) -> pd.DataFrame:
        """获取财务指标 (akshare 支持)."""
        raise NotImplementedError(
            f"{self.source_name} does not support financial indicator fetching"
        )

    # ----- 数据校验 -----

    def validate_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """校验 & 清洗行情数据.

        - 剔除 close 为空的行
        - 剔除 symbol 或 trade_date 为空的行
        - 去重 (symbol + trade_date)
        - volume 负数置 0
        """
        if df.empty:
            return df

        df = df.copy()

        # 必填字段不能为空
        df = df.dropna(subset=["symbol", "trade_date", "close"])

        # 去重
        before = len(df)
        df = df.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        if len(df) < before:
            logger.debug("Removed %d duplicate rows", before - len(df))

        # volume 不能为负
        if "volume" in df.columns:
            df.loc[df["volume"] < 0, "volume"] = 0

        # 数值列: NaN → None (SQLite 兼容)
        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "adj_close", "pre_close", "change_pct", "turnover_rate",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].where(df[col].notna(), None)

        return df

    def validate_fund_nav(self, df: pd.DataFrame) -> pd.DataFrame:
        """校验 & 清洗基金净值数据."""
        if df.empty:
            return df

        df = df.copy()
        df = df.dropna(subset=["code", "nav_date"])
        df = df.drop_duplicates(subset=["code", "nav_date"], keep="last")

        for col in ["unit_nav", "accumulated_nav", "daily_return"]:
            if col in df.columns:
                df[col] = df[col].where(df[col].notna(), None)

        return df
