# data_layer/utils/validators.py
"""数据校验工具 — NaN 处理、类型校验、去重."""

import math
from typing import Any, Optional

import pandas as pd


def safe_float(value: Any) -> Optional[float]:
    """安全转为 float，NaN/Inf → None."""
    if value is None:
        return None
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


def safe_int(value: Any) -> Optional[int]:
    """安全转为 int."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def drop_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """剔除行情 DataFrame 中的无效行.

    规则:
        - close 为空 → 删除
        - high < low → 交换
        - volume < 0 → 置 0
        - 日期重复 → 保留最后一条
    """
    if df.empty:
        return df

    df = df.copy()

    # 必填字段
    if "close" in df.columns:
        df = df.dropna(subset=["close"])

    # 高低价修正
    if "high" in df.columns and "low" in df.columns:
        mask = df["high"] < df["low"]
        df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values

    # 量不能为负
    if "volume" in df.columns:
        df.loc[df["volume"] < 0, "volume"] = 0

    # 去重
    dedup_cols = ["symbol", "trade_date"]
    dup_cols_in_df = [c for c in dedup_cols if c in df.columns]
    if dup_cols_in_df:
        df = df.drop_duplicates(subset=dup_cols_in_df, keep="last")

    return df


def validate_symbol(symbol: str, market: str) -> bool:
    """校验股票代码格式.

    - 美股: 纯字母 1-5 位, e.g. AAPL, BRK-B
    - A股: 纯数字 6 位, e.g. 600519
    """
    if market == "US":
        # 允许字母 + 可选连字符
        return all(c.isalpha() or c == "-" for c in symbol) and 1 <= len(symbol) <= 10
    elif market == "CN":
        return symbol.isdigit() and len(symbol) == 6
    return False
