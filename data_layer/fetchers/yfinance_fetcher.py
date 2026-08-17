# data_layer/fetchers/yfinance_fetcher.py
"""yfinance 数据获取器 — 美股行情."""

import logging
import time
from datetime import date, timedelta
from typing import Optional
import math

import pandas as pd
import yfinance as yf

from data_layer.fetchers.base import AbstractFetcher
from data_layer.models.price import _parse_date
from data_layer.models.stock import Stock

logger = logging.getLogger(__name__)


class YFinanceFetcher(AbstractFetcher):
    """yfinance 美股数据获取器.

    功能:
        - 获取日线 OHLCV (含复权价)
        - 获取股票基础信息 (名称/行业/板块)
        - 支持批量 symbol 下载
    """

    market = "US"
    source_name = "yfinance"

    def __init__(
        self,
        timeout: int = 30,
        retry_count: int = 3,
        retry_delay: int = 5,
    ):
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay

    # ================================================================
    # 行情数据
    # ================================================================

    def fetch_daily_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """批量获取美股日线行情.

        使用 yf.download() 批量下载，自动转换为统一格式。
        """
        if not symbols:
            return pd.DataFrame()

        self._validate_symbols(symbols)

        df = self._download_with_retry(symbols, start_date, end_date)

        if df.empty:
            logger.warning("yfinance returned empty DataFrame for %s", symbols)
            return pd.DataFrame()

        return self._standardize(df)

    def _validate_symbols(self, symbols: list[str]) -> None:
        """校验美股代码格式 (不含 .SS/.SZ 后缀)."""
        for sym in symbols:
            if "." in sym:
                logger.warning(
                    "Symbol '%s' looks like a CN stock code "
                    "(contains '.'). For US stocks, use e.g. 'AAPL'.",
                    sym,
                )

    def _download_with_retry(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """带重试的下载逻辑."""
        last_error = None

        for attempt in range(1, self.retry_count + 1):
            try:
                tickers = yf.Tickers(" ".join(symbols))
                df = tickers.history(
                    start=start_date.isoformat(),
                    end=(end_date + timedelta(days=1)).isoformat(),
                    auto_adjust=False,
                )
                return df if not df.empty else pd.DataFrame()

            except Exception as e:
                last_error = e
                logger.warning(
                    "yfinance download attempt %d/%d failed: %s",
                    attempt, self.retry_count, e,
                )
                if attempt < self.retry_count:
                    time.sleep(self.retry_delay)

        raise RuntimeError(
            f"yfinance download failed after {self.retry_count} attempts "
            f"for {symbols}: {last_error}"
        )

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """将 yfinance 输出转换为统一的 DataFrame 格式.

        yfinance 多 symbol 下载输出 MultiIndex 列:
            (Open, AAPL), (Close, AAPL), ...

        单 symbol 输出普通列: Open, Close, ...
        """
        records = []

        if isinstance(df.columns, pd.MultiIndex):
            # 多 symbol: (列名, symbol)
            price_cols = ["Open", "High", "Low", "Close", "Volume"]
            symbols = df.columns.get_level_values(1).unique()

            for sym in symbols:
                sym_df = pd.DataFrame()
                for col in price_cols:
                    if (col, sym) in df.columns:
                        sym_df[col.lower()] = df[(col, sym)]
                # volume 可能小写
                if ("Volume", sym) in df.columns:
                    sym_df["volume"] = df[("Volume", sym)]

                # adj close
                if ("Adj Close", sym) in df.columns:
                    sym_df["adj_close"] = df[("Adj Close", sym)]

                sym_df["symbol"] = sym
                sym_df = sym_df.reset_index()  # Date → 列

                for _, row in sym_df.iterrows():
                    records.append(self._row_to_record(row, sym))

        else:
            # 单 symbol
            sym = symbols[0] if hasattr(self, '_symbols') else df.index.name
            # 对于单 symbol，列是普通的
            sym_from_arg = None
            for _, row in df.iterrows():
                if sym_from_arg is None:
                    # 无法从 DataFrame 推断 symbol，从传入参数取
                    pass
                records.append(self._row_to_record(row, "UNKNOWN"))

        result = pd.DataFrame(records)
        if result.empty:
            return result

        # 日期排序
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result = result.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        return result

    def _row_to_record(self, row, symbol: str) -> dict:
        """将单行数据转为标准化 dict."""
        # 尝试多种方式获取日期
        trade_date = None
        for key in ("Date", "date", "trade_date"):
            val = row.get(key)
            if val is not None:
                trade_date = _parse_date(val)
                break

        return {
            "symbol": symbol,
            "trade_date": trade_date,
            "open": _safe_float(row.get("open")),
            "high": _safe_float(row.get("high")),
            "low": _safe_float(row.get("low")),
            "close": _safe_float(row.get("close")),
            "volume": _safe_float(row.get("volume")),
            "adj_close": _safe_float(row.get("adj_close")),
            "pre_close": None,
            "change_pct": None,
            "turnover_rate": None,
        }

    # ================================================================
    # 股票信息
    # ================================================================

    def fetch_stock_info(self, symbols: list[str]) -> list[Stock]:
        """获取美股基础信息."""
        stocks = []
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                info = ticker.info or {}

                stock = Stock(
                    symbol=sym,
                    market="US",
                    name=info.get("longName") or info.get("shortName"),
                    exchange=info.get("exchange", "NASDAQ"),
                    sector=info.get("sector"),
                    industry=info.get("industry"),
                    currency=info.get("currency", "USD"),
                )
                stocks.append(stock)
            except Exception as e:
                logger.warning("Failed to fetch info for %s: %s", sym, e)
                # 至少保留一个基础记录
                stocks.append(Stock(symbol=sym, market="US"))

        return stocks


# ----- 工具函数 -----

def _safe_float(value) -> Optional[float]:
    """安全转换为 float."""
    if value is None:
        return None
    try:
        result = float(value)
        return None if math.isnan(result) or math.isinf(result) else result
    except (ValueError, TypeError):
        return None
