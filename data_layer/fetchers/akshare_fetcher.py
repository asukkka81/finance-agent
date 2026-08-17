# data_layer/fetchers/akshare_fetcher.py
"""AKShare 数据获取器 — A股 / 基金 / 宏观指标."""

import logging
import time
from datetime import date, timedelta
from typing import Optional
import math

import pandas as pd

from data_layer.fetchers.base import AbstractFetcher
from data_layer.models.price import _parse_date
from data_layer.models.stock import Stock

logger = logging.getLogger(__name__)


class AKShareFetcher(AbstractFetcher):
    """AKShare A股数据获取器.

    功能:
        - A股日线行情 (前复权)
        - 基金净值
        - 宏观指标 (CPI / PMI / M2 / GDP / LPR)
        - 财务指标 (PE/PB/ROE...)
    """

    market = "CN"
    source_name = "akshare"

    def __init__(
        self,
        timeout: int = 60,
        retry_count: int = 3,
        retry_delay: int = 3,
    ):
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._ak = None  # lazy import

    @property
    def ak(self):
        """延迟导入 akshare — 避免未安装时的 crash."""
        if self._ak is None:
            import akshare as ak
            self._ak = ak
        return self._ak

    # ================================================================
    # 行情数据
    # ================================================================

    def fetch_daily_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """获取 A股日线行情.

        逐个 symbol 调用 akshare.stock_zh_a_hist()。
        """
        if not symbols:
            return pd.DataFrame()

        all_records = []
        for sym in symbols:
            try:
                records = self._fetch_single_stock(sym, start_date, end_date)
                all_records.extend(records)
                logger.debug("Fetched %d records for %s", len(records), sym)
            except Exception as e:
                logger.error("Failed to fetch prices for %s: %s", sym, e)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        return df

    def _fetch_single_stock(
        self, symbol: str, start_date: date, end_date: date
    ) -> list[dict]:
        """获取单只 A股的日线数据."""
        period = "daily"

        df = self._retry_call(
            lambda: self.ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq",  # 前复权
            ),
            f"stock_zh_a_hist({symbol})",
        )

        if df is None or df.empty:
            logger.warning("AKShare returned empty data for %s", symbol)
            return []

        # akshare 列名映射
        col_map = {
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
            "换手率": "turnover_rate",
        }

        df = df.rename(columns=col_map)

        records = []
        for _, row in df.iterrows():
            records.append({
                "symbol": symbol,
                "trade_date": row.get("trade_date"),
                "open": _safe_float_series(row.get("open")),
                "high": _safe_float_series(row.get("high")),
                "low": _safe_float_series(row.get("low")),
                "close": _safe_float_series(row.get("close")),
                "volume": _safe_float_series(row.get("volume")),
                "adj_close": None,  # 前复权 close 即 adj_close
                "pre_close": None,
                "change_pct": _safe_float_series(row.get("change_pct")),
                "turnover_rate": _safe_float_series(row.get("turnover_rate")),
            })

        return records

    # ================================================================
    # 股票信息
    # ================================================================

    def fetch_stock_info(self, symbols: list[str]) -> list[Stock]:
        """获取 A股基础信息."""
        stocks = []

        # 批量获取股票信息
        try:
            stock_info_df = self._retry_call(
                lambda: self.ak.stock_zh_a_spot_em(),
                "stock_zh_a_spot_em() for stock info",
            )
        except Exception as e:
            logger.warning("Failed to fetch A-share stock list: %s", e)
            stock_info_df = None

        for sym in symbols:
            name = sym
            exchange = "UNKNOWN"
            prefix = sym[0] if sym else ""

            if prefix == "6":
                exchange = "SSE"
            elif prefix in ("0", "3", "0", "2"):
                exchange = "SZSE"

            # 从 stock_info_df 查找名称
            if stock_info_df is not None and not stock_info_df.empty:
                col_code = None
                col_name = None
                for c in stock_info_df.columns:
                    if "代码" in str(c):
                        col_code = c
                    if "名称" in str(c):
                        col_name = c

                if col_code and col_name:
                    match = stock_info_df[stock_info_df[col_code] == sym]
                    if not match.empty:
                        name = str(match.iloc[0][col_name])

            stocks.append(Stock(
                symbol=sym,
                market="CN",
                name=name,
                exchange=exchange,
                currency="CNY",
            ))

        return stocks

    # ================================================================
    # 基金净值
    # ================================================================

    def fetch_fund_nav(
        self,
        fund_codes: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """批量获取基金净值数据."""
        all_records = []

        for code in fund_codes:
            try:
                df = self._retry_call(
                    lambda c=code: self.ak.fund_open_fund_info_em(
                        symbol=c, indicator="单位净值走势"
                    ),
                    f"fund_open_fund_info_em({code})",
                )
                if df is None or df.empty:
                    continue

                # 标准化列名
                col_map = {
                    "净值日期": "nav_date",
                    "单位净值": "unit_nav",
                    "累计净值": "accumulated_nav",
                    "日增长率": "daily_return",
                }
                df = df.rename(columns=col_map)

                if "nav_date" not in df.columns:
                    # 尝试其他可能的列名
                    for c in df.columns:
                        if "日期" in str(c):
                            col_map[c] = "nav_date"
                            break
                    df = df.rename(columns=col_map)

                df["code"] = code
                df["nav_date"] = pd.to_datetime(df["nav_date"])

                # 过滤日期范围
                mask = (
                    (df["nav_date"] >= pd.Timestamp(start_date)) &
                    (df["nav_date"] <= pd.Timestamp(end_date))
                )
                df = df[mask]

                for _, row in df.iterrows():
                    all_records.append({
                        "code": code,
                        "nav_date": row.get("nav_date"),
                        "unit_nav": _safe_float_series(row.get("unit_nav")),
                        "accumulated_nav": _safe_float_series(row.get("accumulated_nav")),
                        "daily_return": _safe_float_series(row.get("daily_return")),
                        "subscription": None,
                        "redemption": None,
                    })

            except Exception as e:
                logger.error("Failed to fetch fund NAV for %s: %s", code, e)

        return pd.DataFrame(all_records)

    # ================================================================
    # 宏观指标
    # ================================================================

    def fetch_macro_indicator(self, indicator_name: str) -> pd.DataFrame:
        """获取宏观经济指标.

        支持的指标:
            - cpi: 居民消费价格指数
            - pmi: 制造业采购经理指数
            - m2: 货币供应量
            - gdp: 国内生产总值
            - lpr: 贷款市场报价利率
            - social_financing: 社会融资规模
        """
        indicator_map = {
            "cpi": self._fetch_cpi,
            "pmi": self._fetch_pmi,
            "m2": self._fetch_m2,
            "gdp": self._fetch_gdp,
            "lpr": self._fetch_lpr,
            "social_financing": self._fetch_social_financing,
        }

        fetcher = indicator_map.get(indicator_name.lower())
        if fetcher is None:
            logger.warning(
                "Unsupported indicator: %s. Available: %s",
                indicator_name, list(indicator_map.keys()),
            )
            return pd.DataFrame()

        try:
            return fetcher()
        except Exception as e:
            logger.error(
                "Failed to fetch macro indicator '%s': %s", indicator_name, e
            )
            return pd.DataFrame()

    def _fetch_cpi(self) -> pd.DataFrame:
        """CPI 居民消费价格指数."""
        df = self.ak.macro_china_cpi_monthly()
        return self._standardize_macro_df(df, "cpi", "月度", "日期", "全国-当月")

    def _fetch_pmi(self) -> pd.DataFrame:
        """PMI 制造业采购经理指数."""
        df = self.ak.macro_china_pmi()
        return self._standardize_macro_df(df, "pmi", "月度", "日期", "制造业")

    def _fetch_m2(self) -> pd.DataFrame:
        """M2 货币供应量."""
        df = self.ak.macro_china_money_supply()
        # akshare M2 列名可能变化
        col_map = {}
        for c in df.columns:
            if "日期" in str(c):
                col_map[c] = "pub_date"
            if "M2" in str(c) or "货币和准货币" in str(c):
                col_map[c] = "indicator_value"
        df = df.rename(columns=col_map)
        df["indicator_name"] = "m2"
        df["frequency"] = "月度"
        df["source"] = "PBOC"
        cols = ["indicator_name", "indicator_value", "pub_date", "frequency", "source"]
        return df[[c for c in cols if c in df.columns]]

    def _fetch_gdp(self) -> pd.DataFrame:
        """GDP 国内生产总值."""
        df = self.ak.macro_china_gdp()
        return self._standardize_macro_df(df, "gdp", "季度", "日期", "国内生产总值")

    def _fetch_lpr(self) -> pd.DataFrame:
        """LPR 贷款市场报价利率."""
        df = self.ak.macro_china_lpr()
        col_map = {}
        for c in df.columns:
            if "日期" in str(c):
                col_map[c] = "pub_date"
            if "1年" in str(c) and "变动" not in str(c):
                col_map[c] = "indicator_value"
        df = df.rename(columns=col_map)
        df["indicator_name"] = "lpr_1y"
        df["frequency"] = "月度"
        df["source"] = "PBOC"
        cols = ["indicator_name", "indicator_value", "pub_date", "frequency", "source"]
        return df[[c for c in cols if c in df.columns]]

    def _fetch_social_financing(self) -> pd.DataFrame:
        """社会融资规模."""
        df = self.ak.macro_china_shrzgm()
        return self._standardize_macro_df(
            df, "social_financing", "月度", "月份", "社会融资规模增量"
        )

    def _standardize_macro_df(
        self, df: pd.DataFrame, name: str, frequency: str,
        date_col: str, value_col: str,
    ) -> pd.DataFrame:
        """将 akshare 宏观数据标准化."""
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df = df.rename(columns={date_col: "pub_date", value_col: "indicator_value"})
        df["indicator_name"] = name
        df["frequency"] = frequency
        df["source"] = "akshare"
        df["pub_date"] = pd.to_datetime(df["pub_date"])

        cols = ["indicator_name", "indicator_value", "pub_date", "frequency", "source"]
        result = df[[c for c in cols if c in df.columns]]
        return result.dropna(subset=["indicator_value"])

    # ================================================================
    # 财务指标
    # ================================================================

    def fetch_financial_indicators(
        self, symbols: list[str]
    ) -> pd.DataFrame:
        """获取 A股财务指标 (PE/PB/ROE/营收增速等).

        注意: akshare 的财务指标接口较多，使用 stock_financial_analysis_indicator.
        """
        all_records = []

        for sym in symbols:
            try:
                df = self._retry_call(
                    lambda s=sym: self.ak.stock_financial_analysis_indicator(
                        symbol=s, start_year="2015"
                    ),
                    f"stock_financial_analysis_indicator({sym})",
                )
                if df is None or df.empty:
                    logger.warning("No financial indicators for %s", sym)
                    continue

                # 标准化列名
                col_map = {
                    "日期": "report_date",
                    "净资产收益率(%)": "roe",
                    "总资产净利润率(%)": "roa",
                    "营业利润率(%)": "gross_margin",
                    "成本费用利润率(%)": "net_margin",
                    "营业总收入增长率(%)": "revenue_yoy",
                    "归属母公司净利润增长率(%)": "profit_yoy",
                    "资产负债率(%)": "debt_to_equity",
                    "流动比率(%)": "current_ratio",
                }
                df = df.rename(columns=col_map)

                df["symbol"] = sym
                # 判断报告类型
                if "report_date" in df.columns:
                    df["report_date"] = pd.to_datetime(df["report_date"])
                    df["report_type"] = df["report_date"].apply(_infer_report_type)

                for _, row in df.iterrows():
                    all_records.append({
                        "symbol": sym,
                        "report_date": row.get("report_date"),
                        "report_type": row.get("report_type", "年报"),
                        "roe": _safe_float_series(row.get("roe")),
                        "roa": _safe_float_series(row.get("roa")),
                        "gross_margin": _safe_float_series(row.get("gross_margin")),
                        "net_margin": _safe_float_series(row.get("net_margin")),
                        "revenue_yoy": _safe_float_series(row.get("revenue_yoy")),
                        "profit_yoy": _safe_float_series(row.get("profit_yoy")),
                        "debt_to_equity": _safe_float_series(row.get("debt_to_equity")),
                        "current_ratio": _safe_float_series(row.get("current_ratio")),
                        # yfinance 支持的字段
                        "pe_ttm": _safe_float_series(row.get("pe_ttm")),
                        "pb": _safe_float_series(row.get("pb")),
                        "ps_ttm": _safe_float_series(row.get("ps_ttm")),
                    })

            except Exception as e:
                logger.error(
                    "Failed to fetch financial indicators for %s: %s", sym, e
                )

        return pd.DataFrame(all_records)

    # ================================================================
    # 内部辅助
    # ================================================================

    def _retry_call(self, fn, label: str):
        """带重试的函数调用."""
        last_error = None
        for attempt in range(1, self.retry_count + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                logger.warning(
                    "AKShare call '%s' attempt %d/%d failed: %s",
                    label, attempt, self.retry_count, e,
                )
                if attempt < self.retry_count:
                    time.sleep(self.retry_delay)

        raise RuntimeError(
            f"AKShare call '{label}' failed after {self.retry_count} attempts"
        ) from last_error


# ----- 工具函数 -----

def _safe_float_series(value) -> Optional[float]:
    """安全转换 Series 值为 float."""
    if value is None:
        return None
    try:
        result = float(value)
        return None if math.isnan(result) or math.isinf(result) else result
    except (ValueError, TypeError):
        return None


def _infer_report_type(dt) -> str:
    """根据日期推断报告类型."""
    month = dt.month
    day = dt.day
    if month == 12 and day == 31:
        return "年报"
    elif month == 6 and day == 30:
        return "半年报"
    elif month == 9 and day == 30:
        return "三季报"
    elif month == 3 and day == 31:
        return "一季报"
    else:
        # 按月份回退
        if month <= 3:
            return "一季报"
        elif month <= 6:
            return "半年报"
        elif month <= 9:
            return "三季报"
        else:
            return "年报"
