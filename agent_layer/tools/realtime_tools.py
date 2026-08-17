# agent_layer/tools/realtime_tools.py
"""实时数据工具 — 调用 akshare 金融 API 获取实时行情、指数、资金流向、市场趋势.

与 data_tools.py 不同, 这里的工具直接调用外部 API 而不是查本地数据库,
提供 T+0 实时数据, 适用于盘中决策场景。

数据源: 新浪财经 (via akshare) — 东方财富 API 在海外网络环境下不可用。

2026-08-07: 初始化 — 6 个实时工具, 含内存缓存层
"""

import logging
import time
from datetime import datetime
from typing import Optional

from agent_layer.mcp.registry import ToolRegistry
from agent_layer.mcp.types import ToolRole

logger = logging.getLogger(__name__)

# ================================================================
# 内存缓存层 — stock_zh_a_spot() 需要 ~70s 获取全市场数据，缓存避免重复调用
# ================================================================

_cache: dict[str, tuple[float, any]] = {}  # key → (monotonic_timestamp, data)

# TTL 策略
TTL_SPOT = 60       # 实时行情: 60s — 股价变化快
TTL_INDEX = 60      # 指数行情: 60s
TTL_FLOW = 120      # 资金流向: 120s — 更稳定些
TTL_MARGIN = 300    # 融资融券: 300s — 每天更新
TTL_BOND = 600      # 国债收益率: 600s — 变化缓慢


def _get_cached(key: str, ttl: int, fetcher):
    """带 TTL 的缓存获取.

    Args:
        key: 缓存键.
        ttl: 有效期 (秒).
        fetcher: 无参函数, 在缓存未命中时调用.

    Returns:
        fetcher 的返回值 (缓存的或新获取的).
    """
    now = time.monotonic()
    if key in _cache:
        ts, data = _cache[key]
        if now - ts < ttl:
            return data
    data = fetcher()
    _cache[key] = (now, data)
    return data


def _clear_cache() -> None:
    """清空所有缓存 (调试用)."""
    _cache.clear()


def _get_cache_stats() -> dict:
    """返回缓存统计信息."""
    now = time.monotonic()
    stats = {}
    for key, (ts, data) in _cache.items():
        age = now - ts
        if key.startswith("spot_"):
            ttl = TTL_SPOT
        elif key.startswith("index_"):
            ttl = TTL_INDEX
        elif key.startswith("hsgt_"):
            ttl = TTL_FLOW
        elif key.startswith("margin_"):
            ttl = TTL_MARGIN
        elif key.startswith("bond_"):
            ttl = TTL_BOND
        else:
            ttl = 60
        stats[key] = {"age_seconds": round(age, 1), "ttl": ttl, "fresh": age < ttl}
    return stats


# ================================================================
# 工具注册
# ================================================================


def register_realtime_tools(registry: ToolRegistry) -> None:
    """将实时数据工具注册到 ToolRegistry.

    Args:
        registry: ToolRegistry 实例.

    注册的工具:
        - get_realtime_quote: 单只股票实时报价
        - get_realtime_quotes: 多只股票实时报价
        - get_market_indexes: 主要指数行情
        - get_market_trend: 市场整体涨跌统计
        - get_north_flow: 北向/南向资金流向
        - get_bond_yields: 国债收益率曲线
    """

    # ================================================================
    # Helper: 获取 A 股全市场实时行情 (缓存 ~70s 的 API 调用)
    # ================================================================

    def _fetch_spot_data():
        """获取全市场 A 股实时行情 DataFrame.

        Returns:
            akshare stock_zh_a_spot() 的原始 DataFrame,
            列名已映射为英文.
            如果 akshare 未安装或 API 失败，返回空 DataFrame.
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare not installed, cannot fetch real-time spot data")
            return None

        try:
            df = ak.stock_zh_a_spot()

            if df is None or df.empty:
                logger.warning("stock_zh_a_spot returned empty data")
                return None

            # 清理代码列 — 新浪数据如 'sh600519', 去掉前缀
            df = df.copy()
            if "代码" in df.columns:
                df["代码"] = df["代码"].astype(str).str.replace("sh", "").str.replace("sz", "")

            logger.debug("Fetched %d A-share spot quotes", len(df))
            return df
        except Exception as e:
            logger.error("Failed to fetch spot data: %s", e)
            return None

    def _get_spot_df():
        """获取缓存的 spot DataFrame."""
        return _get_cached("spot_em", TTL_SPOT, _fetch_spot_data)

    def _fetch_index_data():
        """获取全市场指数行情 DataFrame."""
        try:
            import akshare as ak
        except ImportError:
            return None

        try:
            df = ak.stock_zh_index_spot_sina()
            if df is None or df.empty:
                return None
            logger.debug("Fetched %d index quotes", len(df))
            return df
        except Exception as e:
            logger.error("Failed to fetch index data: %s", e)
            return None

    def _get_index_df():
        """获取缓存的 index DataFrame."""
        return _get_cached("index_sina", TTL_INDEX, _fetch_index_data)

    # ================================================================
    # Tool 1: 单只股票实时报价
    # ================================================================

    @registry.register(
        name="get_realtime_quote",
        description=(
            "获取单只 A 股的实时行情数据。返回最新价、涨跌幅、涨跌额、今开、昨收、"
            "最高、最低、成交量、成交额等信息。数据来源: 新浪财经。"
            "适用于: 盘中实时查价、快速了解个股表现。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码，6位数字字符串，如 '600519' (贵州茅台), '000858' (五粮液)",
                },
            },
            "required": ["symbol"],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_realtime_quote(symbol: str) -> dict:
        """获取单只股票实时报价."""
        df = _get_spot_df()
        if df is None:
            return {"error": "无法获取实时行情数据。akshare 可能未安装或 API 不可用。"}

        # 清理输入 — 移除可能的前缀
        clean_symbol = str(symbol).replace("sh", "").replace("sz", "").strip()

        # 精确匹配
        mask = df["代码"] == clean_symbol
        if not mask.any():
            # 尝试模糊匹配 (名称中包含)
            if "名称" in df.columns:
                name_mask = df["名称"].astype(str).str.contains(clean_symbol, na=False)
                if name_mask.any():
                    mask = name_mask
                else:
                    return {
                        "error": f"未找到股票: {symbol}。请确认代码是否正确（6位数字）",
                        "hint": "使用 search_stocks 工具搜索正确的股票代码",
                    }
            else:
                return {
                    "error": f"未找到股票: {symbol}",
                    "hint": "使用 search_stocks 工具搜索正确的股票代码",
                }

        row = df[mask].iloc[0]

        return {
            "symbol": str(row.get("代码", clean_symbol)),
            "name": str(row.get("名称", "")),
            "price": _safe_float(row.get("最新价")),
            "change_pct": _safe_float(row.get("涨跌幅")),
            "change_amount": _safe_float(row.get("涨跌额")),
            "open": _safe_float(row.get("今开")),
            "high": _safe_float(row.get("最高")),
            "low": _safe_float(row.get("最低")),
            "prev_close": _safe_float(row.get("昨收")),
            "volume": _safe_float(row.get("成交量")),
            "amount": _safe_float(row.get("成交额")),
            "timestamp": str(row.get("时间戳", "")),
            "data_source": "新浪财经 (实时)",
            "fetch_time": datetime.now().strftime("%H:%M:%S"),
        }

    # ================================================================
    # Tool 2: 多只股票实时报价
    # ================================================================

    @registry.register(
        name="get_realtime_quotes",
        description=(
            "批量获取多只 A 股的实时行情数据。传入股票代码列表，返回每只股票的最新价、"
            "涨跌幅、成交量、成交额等。适合同时比较多个标的。数据来源: 新浪财经。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "股票代码列表，如 ['600519', '000858', '300750']",
                },
            },
            "required": ["symbols"],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_realtime_quotes(symbols: list[str]) -> dict:
        """批量获取股票实时报价."""
        if not symbols:
            return {"error": "symbols 列表不能为空"}

        df = _get_spot_df()
        if df is None:
            return {"error": "无法获取实时行情数据。akshare 可能未安装或 API 不可用。"}

        # 清理 symbol
        clean_symbols = [str(s).replace("sh", "").replace("sz", "").strip() for s in symbols]

        results = []
        not_found = []

        for original, clean in zip(symbols, clean_symbols):
            mask = df["代码"] == clean
            if mask.any():
                row = df[mask].iloc[0]
                results.append({
                    "symbol": str(row.get("代码", clean)),
                    "name": str(row.get("名称", "")),
                    "price": _safe_float(row.get("最新价")),
                    "change_pct": _safe_float(row.get("涨跌幅")),
                    "change_amount": _safe_float(row.get("涨跌额")),
                    "open": _safe_float(row.get("今开")),
                    "high": _safe_float(row.get("最高")),
                    "low": _safe_float(row.get("最低")),
                    "prev_close": _safe_float(row.get("昨收")),
                    "volume": _safe_float(row.get("成交量")),
                    "amount": _safe_float(row.get("成交额")),
                })
            else:
                not_found.append(original)

        return {
            "count": len(results),
            "results": results,
            "not_found": not_found if not_found else None,
            "data_source": "新浪财经 (实时)",
            "fetch_time": datetime.now().strftime("%H:%M:%S"),
        }

    # ================================================================
    # Tool 3: 主要指数行情
    # ================================================================

    # 用户可能关注的主要指数
    MAJOR_INDEXES = [
        "上证指数", "深证成指", "沪深300", "创业板指",
        "科创50", "上证50", "中证500", "中证1000",
        "恒生指数", "国企指数",
    ]

    @registry.register(
        name="get_market_indexes",
        description=(
            "获取中国主要股票指数的实时行情，包括上证指数、深证成指、沪深300、"
            "创业板指、科创50、中证500、恒生指数等。返回最新点位、涨跌幅、成交量。"
            "适用于: 大盘走势判断、市场整体表现分析。"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_market_indexes() -> dict:
        """获取主要指数实时行情."""
        df = _get_index_df()
        if df is None:
            return {"error": "无法获取指数行情数据。akshare 可能未安装或 API 不可用。"}

        indexes = []
        for _, row in df.iterrows():
            name = str(row.get("名称", ""))
            code = str(row.get("代码", ""))

            # 筛选主要指数
            if not any(kw in name for kw in MAJOR_INDEXES):
                continue

            indexes.append({
                "code": code,
                "name": name,
                "price": _safe_float(row.get("最新价")),
                "change_pct": _safe_float(row.get("涨跌幅")),
                "change_amount": _safe_float(row.get("涨跌额")),
                "open": _safe_float(row.get("今开")),
                "high": _safe_float(row.get("最高")),
                "low": _safe_float(row.get("最低")),
                "prev_close": _safe_float(row.get("昨收")),
                "volume": _safe_float(row.get("成交量")),
                "amount": _safe_float(row.get("成交额")),
            })

        # 按涨跌幅排序 (涨在前)
        indexes.sort(key=lambda x: x["change_pct"] or 0, reverse=True)

        # 市场概览
        up_count = sum(1 for i in indexes if (i["change_pct"] or 0) > 0)
        down_count = sum(1 for i in indexes if (i["change_pct"] or 0) < 0)

        return {
            "indexes": indexes,
            "market_breadth": f"{len(indexes)} 个指数: {up_count}涨 {down_count}跌",
            "data_source": "新浪财经 (实时)",
            "fetch_time": datetime.now().strftime("%H:%M:%S"),
        }

    # ================================================================
    # Tool 4: 市场整体趋势
    # ================================================================

    @registry.register(
        name="get_market_trend",
        description=(
            "获取 A 股市场整体涨跌统计。包括上涨/下跌/平盘家数及比例、"
            "涨停/跌停数量、涨幅榜 Top5、跌幅榜 Top5、全市场成交额等。"
            "适用于: 市场情绪判断、大势分析。"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_market_trend() -> dict:
        """获取市场整体涨跌统计."""
        df = _get_spot_df()
        if df is None:
            return {"error": "无法获取实时行情数据。akshare 可能未安装或 API 不可用。"}

        total = len(df)

        if "涨跌幅" not in df.columns:
            return {"error": "数据格式异常: 缺少涨跌幅列"}

        change_pct = df["涨跌幅"].astype(float)

        up_count = int((change_pct > 0).sum())
        down_count = int((change_pct < 0).sum())
        flat_count = int((change_pct == 0).sum())
        up_ratio = round(up_count / total * 100, 1) if total > 0 else 0

        # 涨停/跌停 (A 股主板 ±10%，科创/创业板 ±20%，近似)
        limit_up_count = int((change_pct >= 9.9).sum())
        limit_down_count = int((change_pct <= -9.9).sum())

        # 涨幅 Top5
        top5 = df.nlargest(5, "涨跌幅")
        top_gainers = []
        for _, r in top5.iterrows():
            top_gainers.append({
                "symbol": str(r.get("代码", "")),
                "name": str(r.get("名称", "")),
                "price": _safe_float(r.get("最新价")),
                "change_pct": _safe_float(r.get("涨跌幅")),
            })

        # 跌幅 Top5
        bottom5 = df.nsmallest(5, "涨跌幅")
        top_losers = []
        for _, r in bottom5.iterrows():
            top_losers.append({
                "symbol": str(r.get("代码", "")),
                "name": str(r.get("名称", "")),
                "price": _safe_float(r.get("最新价")),
                "change_pct": _safe_float(r.get("涨跌幅")),
            })

        # 全市场成交额
        total_amount = 0.0
        if "成交额" in df.columns:
            total_amount = df["成交额"].astype(float).sum()

        return {
            "total_stocks": total,
            "up_count": up_count,
            "down_count": down_count,
            "flat_count": flat_count,
            "up_ratio": up_ratio,
            "limit_up_count": limit_up_count,
            "limit_down_count": limit_down_count,
            "top_gainers": top_gainers,
            "top_losers": top_losers,
            "total_amount": round(total_amount, 2),
            "total_amount_display": f"{total_amount / 1e12:.2f} 万亿" if total_amount > 0 else "N/A",
            "data_source": "新浪财经 (实时)",
            "fetch_time": datetime.now().strftime("%H:%M:%S"),
        }

    # ================================================================
    # Tool 5: 北向/南向资金流向
    # ================================================================

    def _fetch_north_flow():
        """获取沪深港通资金流向."""
        try:
            import akshare as ak
        except ImportError:
            return None
        try:
            df = ak.stock_hsgt_fund_flow_summary_em()
            if df is None or df.empty:
                return None
            return df
        except Exception as e:
            logger.error("Failed to fetch north flow: %s", e)
            return None

    @registry.register(
        name="get_north_flow",
        description=(
            "获取沪深港通资金流向汇总（北向资金 + 南向资金）。"
            "包括当日成交净买额、资金净流入、当日资金余额、上涨/下跌数等。"
            "适用于: 外资动向分析、市场资金面判断。"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_north_flow() -> dict:
        """获取北向/南向资金流向."""
        df = _get_cached("hsgt_flow", TTL_FLOW, _fetch_north_flow)
        if df is None:
            return {"error": "无法获取资金流向数据。akshare 可能未安装或 API 不可用。"}

        # 列名映射
        records = []
        for _, row in df.iterrows():
            direction = str(row.get("资金方向", ""))
            flow_type = str(row.get("板块", ""))
            records.append({
                "date": str(row.get("交易日", "")),
                "market": flow_type,
                "direction": direction,
                "net_buy": _safe_float(row.get("成交净买额")),
                "net_inflow": _safe_float(row.get("资金净流入")),
                "balance": _safe_float(row.get("当日资金余额")),
                "up_count": _safe_int(row.get("上涨数")),
                "flat_count": _safe_int(row.get("持平数")),
                "down_count": _safe_int(row.get("下跌数")),
            })

        # 按方向分组摘要
        north_records = [r for r in records if r["direction"] == "北向"]
        south_records = [r for r in records if r["direction"] == "南向"]

        summary_parts = []
        for r in north_records:
            if r["net_buy"] is not None:
                direction_text = "流入" if r["net_buy"] > 0 else "流出"
                summary_parts.append(
                    f"{r['market']}: {direction_text} {abs(r['net_buy']):.2f} 亿"
                )

        return {
            "records": records,
            "north_bound": north_records,
            "south_bound": south_records,
            "summary": " | ".join(summary_parts) if summary_parts else "",
            "data_source": "东方财富 (沪深港通)",
            "fetch_time": datetime.now().strftime("%H:%M:%S"),
        }

    # ================================================================
    # Tool 6: 国债收益率曲线
    # ================================================================

    def _fetch_bond_yields():
        """获取国债收益率曲线."""
        try:
            import akshare as ak
        except ImportError:
            return None
        try:
            df = ak.bond_china_yield()
            if df is None or df.empty:
                return None
            return df
        except Exception as e:
            logger.error("Failed to fetch bond yields: %s", e)
            return None

    @registry.register(
        name="get_bond_yields",
        description=(
            "获取中国国债收益率曲线最新数据。包括 3 月、6 月、1 年、3 年、5 年、"
            "7 年、10 年、30 年期国债收益率。"
            "适用于: 宏观利率环境判断、无风险利率参考、债券市场分析。"
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_bond_yields() -> dict:
        """获取国债收益率曲线."""
        df = _get_cached("bond_yield", TTL_BOND, _fetch_bond_yields)
        if df is None:
            return {"error": "无法获取国债收益率数据。akshare 可能未安装或 API 不可用。"}

        # 取最新一行 (国债收益率曲线)
        govt_bonds = df[df["曲线名称"].str.contains("国债", na=False)]
        if govt_bonds.empty:
            # fallback: 取最后一行
            row = df.iloc[-1]
        else:
            row = govt_bonds.iloc[-1]

        tenors = [
            ("3月", "3M"), ("6月", "6M"), ("1年", "1Y"),
            ("3年", "3Y"), ("5年", "5Y"), ("7年", "7Y"),
            ("10年", "10Y"), ("30年", "30Y"),
        ]

        yields = []
        for cn_name, en_name in tenors:
            val = _safe_float(row.get(cn_name))
            if val is not None:
                yields.append({"tenor": en_name, "tenor_cn": cn_name, "yield": val})

        return {
            "date": str(row.get("日期", "")),
            "curve_name": str(row.get("曲线名称", "中债国债收益率曲线")),
            "yields": yields,
            "data_source": "中国债券信息网 (via akshare)",
            "fetch_time": datetime.now().strftime("%H:%M:%S"),
        }

    # ================================================================
    # 注册完成
    # ================================================================

    logger.info(
        "Registered %d realtime tools: %s",
        6,
        [
            "get_realtime_quote",
            "get_realtime_quotes",
            "get_market_indexes",
            "get_market_trend",
            "get_north_flow",
            "get_bond_yields",
        ],
    )


# ================================================================
# 辅助函数
# ================================================================

def _safe_float(value) -> Optional[float]:
    """安全地把值转为 float，失败返回 None."""
    if value is None:
        return None
    try:
        result = float(value)
        import math
        if math.isnan(result) or math.isinf(result):
            return None
        return round(result, 4)
    except (ValueError, TypeError):
        # 尝试处理带前缀的字符串如 'sh600519'
        return None


def _safe_int(value) -> Optional[int]:
    """安全地把值转为 int，失败返回 None."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None
