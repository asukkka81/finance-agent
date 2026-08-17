# data_layer/services/data_service.py
"""核心数据服务 — 编排 Fetcher + Repository，提供统一数据接口.

这是数据层的主要 API 入口，也是后续 MCP tool 的 handler 所在地。
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from data_layer.config import Config, load_config
from data_layer.database.connection import DatabaseManager
from data_layer.fetchers.base import AbstractFetcher
from data_layer.fetchers.fetcher_factory import FetcherFactory
from data_layer.repositories.fund_repository import FundRepository
from data_layer.repositories.macro_repository import MacroRepository
from data_layer.repositories.price_repository import PriceRepository
from data_layer.repositories.stock_repository import StockRepository

logger = logging.getLogger(__name__)


# ================================================================
# 数据传输对象
# ================================================================

@dataclass
class SyncResult:
    """数据同步结果."""

    symbol: str
    market: str
    start_date: date
    end_date: date
    records_fetched: int = 0
    records_inserted: int = 0
    records_skipped: int = 0
    status: str = "success"  # success / failed / partial
    error: Optional[str] = None
    duration_seconds: float = 0.0


# ================================================================
# 核心服务
# ================================================================

class DataService:
    """核心数据服务.

    桥接 Fetcher (外部数据源) 和 Repository (本地数据库)，
    提供同步、查询、搜索三大类操作。

    Usage::

        config = load_config()
        db = DatabaseManager(config.db_path)
        db.initialize_schema()

        svc = DataService(db, config)
        # 同步美股行情
        results = svc.sync_stock_prices(
            symbols=["AAPL", "MSFT"], market="US",
            start=date(2024, 1, 1), end=date(2024, 12, 31),
        )
        # 查询行情
        df = svc.get_price_history("AAPL", date(2024, 1, 1), date.today())
    """

    def __init__(self, db: DatabaseManager, config: Optional[Config] = None):
        self.db = db
        self.config = config or load_config()
        self._factory = FetcherFactory(self.config)

        # 初始化仓库
        self.stock_repo = StockRepository(db)
        self.price_repo = PriceRepository(db)
        self.fund_repo = FundRepository(db)
        self.macro_repo = MacroRepository(db)

    # ================================================================
    # 行情数据同步
    # ================================================================

    def sync_stock_prices(
        self,
        symbols: list[str],
        market: str,
        start: date,
        end: date,
        incremental: bool = True,
    ) -> list[SyncResult]:
        """股票行情同步 — 核心方法.

        流程:
            1. 确保 stock 元数据已入库
            2. Fetcher 拉取数据
            3. 校验 & 清洗
            4. Repository 批量 upsert
            5. 返回 SyncResult (含统计信息)

        Args:
            symbols: 股票代码列表.
            market: 'US' | 'CN'.
            start: 开始日期.
            end: 结束日期.
            incremental: True → 从数据库最新日期增量拉取;
                        False → 全量覆盖指定区间.

        Returns:
            SyncResult 列表 (每个 symbol 一个).
        """
        fetcher = self._factory.get_by_market(market)

        results = []
        for symbol in symbols:
            t0 = datetime.now()
            try:
                actual_start = start

                if incremental:
                    # 查最新日期，从后一天开始增量拉取
                    stock = self.stock_repo.find_by_symbol(symbol)
                    if stock:
                        latest = self.price_repo.get_latest_trade_date(stock.id)
                        if latest and latest >= start:
                            actual_start = latest + timedelta(days=1)

                if actual_start > end:
                    results.append(SyncResult(
                        symbol=symbol, market=market,
                        start_date=actual_start, end_date=end,
                        status="success", records_fetched=0,
                        records_inserted=0, records_skipped=0,
                        duration_seconds=(datetime.now() - t0).total_seconds(),
                    ))
                    continue

                # Step 1: 确保 stock 元数据
                stock_id = self.stock_repo.get_or_create(
                    symbol, market,
                    name=self._fetch_stock_name(fetcher, symbol),
                )

                # Step 2: Fetcher 拉取
                df = fetcher.fetch_daily_prices([symbol], actual_start, end)

                if df.empty:
                    results.append(SyncResult(
                        symbol=symbol, market=market,
                        start_date=actual_start, end_date=end,
                        status="success", records_fetched=0,
                        duration_seconds=(datetime.now() - t0).total_seconds(),
                    ))
                    continue

                # Step 3: 校验 & 清洗
                df = fetcher.validate_prices(df)
                if df.empty:
                    results.append(SyncResult(
                        symbol=symbol, market=market,
                        start_date=actual_start, end_date=end,
                        status="success", records_fetched=0,
                        duration_seconds=(datetime.now() - t0).total_seconds(),
                    ))
                    continue

                # Step 4: 批量写入
                stat = self.price_repo.upsert_prices_from_df(
                    df, {symbol: stock_id},
                )

                duration = (datetime.now() - t0).total_seconds()
                results.append(SyncResult(
                    symbol=symbol, market=market,
                    start_date=actual_start, end_date=end,
                    records_fetched=len(df),
                    records_inserted=stat["inserted"],
                    records_skipped=stat["skipped"],
                    status="success",
                    duration_seconds=duration,
                ))

                logger.info(
                    "Synced %s: %d fetched, %d inserted, %d skipped (%.1fs)",
                    symbol, len(df), stat["inserted"], stat["skipped"], duration,
                )

            except Exception as e:
                duration = (datetime.now() - t0).total_seconds()
                logger.error("Sync failed for %s: %s", symbol, e)
                results.append(SyncResult(
                    symbol=symbol, market=market,
                    start_date=start, end_date=end,
                    status="failed", error=str(e),
                    duration_seconds=duration,
                ))

        return results

    def sync_all_us(
        self,
        start: date,
        end: date,
        incremental: bool = True,
    ) -> list[SyncResult]:
        """同步配置中所有美股行情."""
        return self.sync_stock_prices(
            self.config.us_symbols, "US", start, end, incremental,
        )

    def sync_all_cn(
        self,
        start: date,
        end: date,
        incremental: bool = True,
    ) -> list[SyncResult]:
        """同步配置中所有 A股行情."""
        return self.sync_stock_prices(
            self.config.cn_symbols, "CN", start, end, incremental,
        )

    # ================================================================
    # 查询接口 (MCP tool handler 入口)
    # ================================================================

    def get_price_history(
        self, symbol: str, start: date, end: date
    ) -> pd.DataFrame:
        """获取单只股票的历史行情 (DataFrame)."""
        stock = self.stock_repo.find_by_symbol(symbol)
        if not stock:
            return pd.DataFrame()

        df = self.price_repo.find_history_as_df(stock.id, start, end)
        if not df.empty:
            df["symbol"] = symbol
        return df

    def get_latest_prices(self, market: str) -> pd.DataFrame:
        """获取某市场所有股票的最新行情."""
        stocks = self.stock_repo.find_by_market(market)
        if not stocks:
            return pd.DataFrame()

        stock_ids = [s.id for s in stocks]
        df = self.price_repo.find_latest_batch(stock_ids, days=1)

        # 只保留每只股票的最新一条
        if not df.empty:
            df = df.sort_values(
                ["stock_id", "trade_date"]
            ).groupby("stock_id").last().reset_index()

        return df

    def search_stocks(
        self, keyword: str, limit: int = 20
    ) -> list[dict]:
        """模糊搜索股票."""
        stocks = self.stock_repo.search(keyword, limit)
        return [
            {
                "symbol": s.symbol,
                "name": s.name,
                "market": s.market,
                "exchange": s.exchange,
                "sector": s.sector,
            }
            for s in stocks
        ]

    # ================================================================
    # 基金数据
    # ================================================================

    def sync_fund_nav(
        self,
        fund_codes: list[str],
        start: date,
        end: date,
    ) -> list[SyncResult]:
        """同步基金净值."""
        fetcher = self._factory.get("akshare")
        results = []

        for code in fund_codes:
            t0 = datetime.now()
            try:
                fund_id = self.fund_repo.get_or_create(code)

                df = fetcher.fetch_fund_nav([code], start, end)
                if df.empty:
                    results.append(SyncResult(
                        symbol=code, market="CN",
                        start_date=start, end_date=end,
                        status="success",
                        duration_seconds=(datetime.now() - t0).total_seconds(),
                    ))
                    continue

                df = fetcher.validate_fund_nav(df)

                # 写入 fund_nav 表
                nav_list = []
                for _, row in df.iterrows():
                    from data_layer.models.fund import FundNAV
                    from data_layer.models.price import _parse_date, _optional_float

                    nav_date = row.get("nav_date")
                    nav_list.append(FundNAV(
                        fund_id=fund_id,
                        nav_date=_parse_date(nav_date) if nav_date else date.today(),
                        unit_nav=_optional_float(row.get("unit_nav")),
                        accumulated_nav=_optional_float(row.get("accumulated_nav")),
                        daily_return=_optional_float(row.get("daily_return")),
                    ))

                inserted = self.fund_repo.upsert_nav_batch(nav_list)
                duration = (datetime.now() - t0).total_seconds()

                results.append(SyncResult(
                    symbol=code, market="CN",
                    start_date=start, end_date=end,
                    records_fetched=len(df), records_inserted=inserted,
                    records_skipped=len(df) - inserted,
                    status="success", duration_seconds=duration,
                ))

            except Exception as e:
                results.append(SyncResult(
                    symbol=code, market="CN",
                    start_date=start, end_date=end,
                    status="failed", error=str(e),
                    duration_seconds=(datetime.now() - t0).total_seconds(),
                ))

        return results

    # ================================================================
    # 宏观指标
    # ================================================================

    def sync_macro_indicators(self) -> list[SyncResult]:
        """同步配置中所有宏观指标."""
        fetcher = self._factory.get("akshare")
        results = []

        for indicator_name in self.config.macro_indicators:
            t0 = datetime.now()
            try:
                df = fetcher.fetch_macro_indicator(indicator_name)
                if df.empty:
                    continue

                from data_layer.models.macro import MacroIndicator
                from data_layer.models.price import _parse_date, _optional_float

                indicators = []
                for _, row in df.iterrows():
                    pub_date = row.get("pub_date")
                    indicators.append(MacroIndicator(
                        indicator_name=row.get("indicator_name", indicator_name),
                        pub_date=_parse_date(pub_date) if pub_date else date.today(),
                        indicator_value=_optional_float(row.get("indicator_value")),
                        frequency=row.get("frequency"),
                        source=row.get("source", "akshare"),
                    ))

                inserted = self.macro_repo.upsert_indicators(indicators)
                duration = (datetime.now() - t0).total_seconds()

                results.append(SyncResult(
                    symbol=indicator_name, market="CN",
                    start_date=date.today(), end_date=date.today(),
                    records_fetched=len(df), records_inserted=inserted,
                    status="success", duration_seconds=duration,
                ))

            except Exception as e:
                logger.error("Failed to sync macro indicator %s: %s", indicator_name, e)
                results.append(SyncResult(
                    symbol=indicator_name, market="CN",
                    start_date=date.today(), end_date=date.today(),
                    status="failed", error=str(e),
                ))

        return results

    # ================================================================
    # 内部辅助
    # ================================================================

    def _fetch_stock_name(
        self, fetcher: AbstractFetcher, symbol: str
    ) -> Optional[str]:
        """获取单只股票的名称."""
        try:
            stocks = fetcher.fetch_stock_info([symbol])
            if stocks and stocks[0].name:
                return stocks[0].name
        except Exception:
            pass
        return None
