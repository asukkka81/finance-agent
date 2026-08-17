# data_layer/utils/date_utils.py
"""日期工具函数 — 交易日判断、日期范围生成."""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# A股市场休市日期速查 (粗略版 — 每年更新)
# 实际使用时建议从 akshare.tool_trade_date_hist_sina() 动态获取
_CN_HOLIDAYS: set[date] = set()


def get_trade_date_range(
    start: date, end: date, market: str = "US"
) -> list[date]:
    """生成指定日期范围内的所有交易日列表.

    Args:
        start: 开始日期.
        end: 结束日期.
        market: 'US' (周一～周五) 或 'CN' (排除 A股节假日).

    Returns:
        交易日列表.
    """
    if market == "US":
        # 美股: 简单排除周末 (精确假日需纽约证券交易所日历)
        dates = pd.bdate_range(start=start, end=end, freq="C", weekmask="Mon Tue Wed Thu Fri")
        return [d.date() for d in dates]
    else:
        # A股: 排除周末 + 粗略假日
        dates = pd.bdate_range(start=start, end=end, freq="C", weekmask="Mon Tue Wed Thu Fri")
        result = []
        for d in dates:
            d_date = d.date()
            if d_date not in _CN_HOLIDAYS:
                result.append(d_date)
        return result


def is_trade_day(d: date, market: str = "US") -> bool:
    """判断某天是否为交易日.

    Args:
        d: 待判断的日期.
        market: 'US' 或 'CN'.

    Returns:
        True 如果是交易日.
    """
    if d.weekday() >= 5:  # 周六日
        return False
    if market == "CN" and d in _CN_HOLIDAYS:
        return False
    return True


def nearest_trade_day(d: date, market: str = "US") -> date:
    """获取最近的一个交易日 (如果当天不是交易日则往前查找).

    Args:
        d: 参考日期.
        market: 'US' 或 'CN'.

    Returns:
        最近的交易日.
    """
    max_lookback = 10
    for i in range(max_lookback):
        candidate = d - timedelta(days=i)
        if is_trade_day(candidate, market):
            return candidate

    # 兜底: 直接返回参考日期
    return d
