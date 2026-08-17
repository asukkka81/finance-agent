# data_layer/fetchers/fetcher_factory.py
"""Fetcher 工厂 — 根据市场/数据源创建对应的 Fetcher 实例."""

import logging
from typing import Optional

from data_layer.fetchers.base import AbstractFetcher
from data_layer.fetchers.yfinance_fetcher import YFinanceFetcher
from data_layer.fetchers.akshare_fetcher import AKShareFetcher

logger = logging.getLogger(__name__)


class FetcherFactory:
    """Fetcher 工厂.

    支持:
        - 按市场创建: market='US' → YFinanceFetcher, market='CN' → AKShareFetcher
        - 按名称创建: name='yfinance' → YFinanceFetcher, name='akshare' → AKShareFetcher
        - 创建后自动缓存，避免重复初始化

    Usage::

        factory = FetcherFactory(config)
        fetcher = factory.get_by_market("US")   # → YFinanceFetcher
        fetcher = factory.get("yfinance")        # → YFinanceFetcher
    """

    def __init__(self, config):
        """初始化工厂.

        Args:
            config: Config 实例 (data_layer.config.Config).
        """
        self._config = config
        self._cache: dict[str, AbstractFetcher] = {}

    def get_by_market(self, market: str) -> AbstractFetcher:
        """根据市场标识获取 Fetcher.

        Args:
            market: 'US' → YFinanceFetcher, 'CN' → AKShareFetcher.

        Returns:
            AbstractFetcher 实例.

        Raises:
            ValueError: 不支持的市场.
        """
        if market == "US":
            return self.get("yfinance")
        elif market == "CN":
            return self.get("akshare")
        else:
            raise ValueError(
                f"Unsupported market: '{market}'. Supported: US, CN."
            )

    def get(self, name: str) -> AbstractFetcher:
        """根据数据源名称获取 Fetcher (带缓存).

        Args:
            name: 'yfinance' | 'akshare'.

        Returns:
            AbstractFetcher 实例.
        """
        if name in self._cache:
            return self._cache[name]

        fetcher = self._create(name)
        self._cache[name] = fetcher
        logger.debug("Created fetcher: %s", name)
        return fetcher

    def _create(self, name: str) -> AbstractFetcher:
        """创建 Fetcher 实例."""
        if name == "yfinance":
            yf_cfg = self._config.yfinance_config
            return YFinanceFetcher(
                timeout=yf_cfg.get("timeout", 30),
                retry_count=yf_cfg.get("retry_count", 3),
                retry_delay=yf_cfg.get("retry_delay", 5),
            )

        elif name == "akshare":
            ak_cfg = self._config.akshare_config
            return AKShareFetcher(
                timeout=ak_cfg.get("timeout", 60),
                retry_count=ak_cfg.get("retry_count", 3),
                retry_delay=ak_cfg.get("retry_delay", 3),
            )

        else:
            raise ValueError(
                f"Unknown fetcher: '{name}'. Supported: yfinance, akshare."
            )

    def clear_cache(self) -> None:
        """清除 Fetcher 缓存."""
        self._cache.clear()

    @property
    def available_fetchers(self) -> list[str]:
        """列出所有可用的 Fetcher 名称."""
        return ["yfinance", "akshare"]
