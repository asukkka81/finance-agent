# data_layer/services/sync_service.py
"""定时同步服务 — 批量 & 增量数据更新调度."""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from data_layer.config import Config
from data_layer.database.connection import DatabaseManager
from data_layer.services.data_service import DataService, SyncResult

logger = logging.getLogger(__name__)


class SyncService:
    """定时同步服务.

    封装常见的批量同步场景:
        - 首次全量同步 (最近 N 年)
        - 每日增量更新 (默认拉取最近 7 天)
        - 按市场分别同步

    Usage::

        sync = SyncService(db, config)
        # 首次全量同步
        sync.full_sync(years=5)
        # 每日增量更新
        sync.daily_update()
    """

    def __init__(self, db: DatabaseManager, config: Optional[Config] = None):
        self.db = db
        self.config = config or Config({})
        self.service = DataService(db, config)

    # ================================================================
    # 全量同步
    # ================================================================

    def full_sync(self, years: int = 5) -> dict:
        """首次全量同步 — 拉取所有默认股票的历史数据.

        Args:
            years: 拉取多少年的历史数据.

        Returns:
            dict: {'us': [...], 'cn': [...], 'funds': [...], 'macro': [...]}
        """
        end = date.today()
        start = end.replace(year=end.year - years)

        logger.info("Starting full sync: %s → %s (%d years)", start, end, years)

        results = {
            "us": [],
            "cn": [],
            "funds": [],
            "macro": [],
        }

        # US stocks
        if self.config.us_symbols:
            logger.info("Syncing US stocks...")
            results["us"] = self.service.sync_stock_prices(
                self.config.us_symbols, "US", start, end, incremental=False,
            )

        # CN stocks
        if self.config.cn_symbols:
            logger.info("Syncing CN stocks...")
            results["cn"] = self.service.sync_stock_prices(
                self.config.cn_symbols, "CN", start, end, incremental=False,
            )

        # Funds
        if self.config.fund_codes:
            logger.info("Syncing fund NAV...")
            results["funds"] = self.service.sync_fund_nav(
                self.config.fund_codes, start, end,
            )

        # Macro indicators
        logger.info("Syncing macro indicators...")
        results["macro"] = self.service.sync_macro_indicators()

        self._log_summary(results)
        return results

    # ================================================================
    # 每日增量更新
    # ================================================================

    def daily_update(self) -> dict:
        """每日增量更新 — 只拉取最近数据.

        默认增量模式: 从数据库最新日期开始拉取到今日.
        """
        end = date.today()
        start = end - timedelta(days=7)  # 兜底: 最近 7 天

        logger.info("Starting daily update: %s → %s", start, end)

        results = {
            "us": [],
            "cn": [],
            "funds": [],
            "macro": [],
        }

        if self.config.us_symbols:
            results["us"] = self.service.sync_stock_prices(
                self.config.us_symbols, "US", start, end, incremental=True,
            )

        if self.config.cn_symbols:
            results["cn"] = self.service.sync_stock_prices(
                self.config.cn_symbols, "CN", start, end, incremental=True,
            )

        if self.config.fund_codes:
            results["funds"] = self.service.sync_fund_nav(
                self.config.fund_codes, start, end,
            )

        results["macro"] = self.service.sync_macro_indicators()

        self._log_summary(results)
        return results

    # ================================================================
    # 按日期范围同步
    # ================================================================

    def sync_range(
        self,
        start: date,
        end: date,
        markets: Optional[list[str]] = None,
    ) -> dict:
        """按指定日期范围同步所有市场数据."""
        if markets is None:
            markets = ["US", "CN"]

        results = {}
        for market in markets:
            symbols = (
                self.config.us_symbols if market == "US"
                else self.config.cn_symbols
            )
            results[market] = self.service.sync_stock_prices(
                symbols, market, start, end, incremental=False,
            )

        self._log_summary(results)
        return results

    # ================================================================
    # 辅助
    # ================================================================

    def _log_summary(self, results: dict) -> None:
        """打印同步摘要."""
        total_inserted = 0
        total_failed = 0

        for category, items in results.items():
            cat_inserted = sum(
                r.records_inserted for r in items if r.status == "success"
            )
            cat_failed = sum(1 for r in items if r.status == "failed")
            total_inserted += cat_inserted
            total_failed += cat_failed
            logger.info(
                "  %s: %d inserted, %d failed",
                category, cat_inserted, cat_failed,
            )

        logger.info(
            "Sync complete: %d total inserted, %d total failed",
            total_inserted, total_failed,
        )
