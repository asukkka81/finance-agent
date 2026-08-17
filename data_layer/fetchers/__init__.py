# data_layer/fetchers/__init__.py
"""数据获取层 — 从外部数据源抓取金融数据."""

from data_layer.fetchers.base import AbstractFetcher
from data_layer.fetchers.yfinance_fetcher import YFinanceFetcher
from data_layer.fetchers.akshare_fetcher import AKShareFetcher
from data_layer.fetchers.fetcher_factory import FetcherFactory

__all__ = [
    "AbstractFetcher",
    "YFinanceFetcher",
    "AKShareFetcher",
    "FetcherFactory",
]
