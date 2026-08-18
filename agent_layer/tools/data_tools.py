# agent_layer/tools/data_tools.py
"""实时数据查询工具 — 封装 data_layer Service 为 MCP 工具.

对应 Agent 五大工具之: 实时数据查询工具
功能: 股票行情、基金净值、宏观指标、财务指标查询
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from agent_layer.mcp.registry import ToolRegistry
from agent_layer.mcp.types import ToolRole

logger = logging.getLogger(__name__)


def register_data_tools(
    registry: ToolRegistry,
    data_service=None,  # DataService from data_layer
    stock_repo=None,    # StockRepository
    price_repo=None,    # PriceRepository
    fund_repo=None,     # FundRepository
    macro_repo=None,    # MacroRepository
) -> None:
    """将数据层功能注册为 MCP 工具.

    Args:
        registry: ToolRegistry 实例.
        data_service: DataService 实例 (如未提供，使用独立的 Repository).
        stock_repo, price_repo, fund_repo, macro_repo: 各 Repository.
    """

    # ----- 股票行情 -----

    @registry.register(
        name="get_stock_price",
        description=(
            "获取指定股票在日期范围内的日线行情数据 (OHLCV)。"
            "返回开盘价、最高价、最低价、收盘价、成交量、涨跌幅等信息。"
            "适用于: 个股走势分析、历史价格查询、技术指标计算。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码。美股如 AAPL, MSFT; A股如 600519 (贵州茅台), 000858 (五粮液)",
                },
                "start": {
                    "type": "string",
                    "description": "开始日期，格式 YYYY-MM-DD，如 2024-01-01",
                },
                "end": {
                    "type": "string",
                    "description": "结束日期，格式 YYYY-MM-DD，如 2024-12-31。默认为今天",
                },
            },
            "required": ["symbol", "start"],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_stock_price(symbol: str, start: str, end: Optional[str] = None) -> dict:
        """获取股票日线行情."""
        if end is None:
            end = date.today().isoformat()

        try:
            start_date = datetime.strptime(start, "%Y-%m-%d").date()
            end_date = datetime.strptime(end, "%Y-%m-%d").date()
        except ValueError:
            return {"error": f"日期格式错误: start={start}, end={end}。请使用 YYYY-MM-DD 格式"}

        if not price_repo or not stock_repo:
            return {"error": "数据仓库未初始化，无法查询行情"}

        stock = stock_repo.find_by_symbol(symbol)
        if not stock:
            return {
                "error": f"未找到股票: {symbol}。请使用 search_stocks 搜索正确的股票代码",
                "suggestion": f"尝试 search_stocks(keyword='{symbol}') 查找",
            }

        prices = price_repo.find_history(stock.id, start_date, end_date)

        if not prices:
            return {
                "symbol": symbol,
                "name": stock.name,
                "market": stock.market,
                "period": f"{start} ~ {end}",
                "count": 0,
                "message": f"该时间段内无行情数据。可能需要先同步数据: sync_stock_prices('{symbol}', '{start}', '{end}')",
            }

        # 返回结构化数据 + 简要统计
        price_data = []
        for p in prices:
            price_data.append({
                "date": p.trade_date.isoformat(),
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume,
                "change_pct": p.change_pct,
            })

        closes = [p.close for p in prices if p.close]
        summary = {
            "symbol": symbol,
            "name": stock.name,
            "market": stock.market,
            "period": f"{start} ~ {end}",
            "count": len(price_data),
            "latest_close": closes[-1] if closes else None,
            "max_high": max((p.high for p in prices if p.high), default=None),
            "min_low": min((p.low for p in prices if p.low), default=None),
            "avg_volume": sum((p.volume for p in prices if p.volume), 0) / max(len(prices), 1),
        }

        return {
            "summary": summary,
            "data": price_data[:30],  # 最多返回 30 条，避免上下文溢出
            "note": f"共 {len(price_data)} 条记录，返回最近 30 条" if len(price_data) > 30 else "",
        }

    # ----- 股票搜索 -----

    @registry.register(
        name="search_stocks",
        description=(
            "按关键词搜索股票。支持按名称或代码模糊搜索。"
            "适用于: 用户只知道公司名称但不知道股票代码时。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "搜索关键词，如 '茅台'、'Apple'、'新能源'",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回数量上限，默认 10",
                },
            },
            "required": ["keyword"],
        },
        role=ToolRole.DATA_QUERY,
    )
    def search_stocks_tool(keyword: str, limit: int = 10) -> dict:
        """搜索股票."""
        if not stock_repo:
            return {"error": "数据仓库未初始化"}

        stocks = stock_repo.search(keyword, limit=limit)
        results = []
        for s in stocks:
            results.append({
                "symbol": s.symbol,
                "name": s.name,
                "market": "A股" if s.market == "CN" else "美股",
                "exchange": s.exchange,
                "sector": s.sector,
            })

        return {
            "keyword": keyword,
            "count": len(results),
            "results": results,
        }

    # ----- 宏观指标 -----

    @registry.register(
        name="get_macro_indicator",
        description=(
            "获取宏观经济指标数据。支持: CPI(居民消费价格指数)、PMI(制造业采购经理指数)、"
            "M2(货币供应量)、GDP(国内生产总值)、LPR(贷款市场报价利率)。"
            "适用于: 宏观经济分析、政策影响评估。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "indicator_name": {
                    "type": "string",
                    "description": "指标名称: cpi / pmi / m2 / gdp / lpr",
                },
            },
            "required": ["indicator_name"],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_macro_indicator(indicator_name: str) -> dict:
        """获取宏观指标."""
        if not macro_repo:
            return {"error": "数据仓库未初始化"}

        end = date.today()
        start = end.replace(year=end.year - 2)  # 最近 2 年

        indicators = macro_repo.find_indicator(indicator_name.lower(), start, end)

        if not indicators:
            return {
                "indicator": indicator_name,
                "count": 0,
                "message": "该指标暂无数据，可能需要先同步宏观数据",
            }

        data = [
            {"date": i.pub_date.isoformat(), "value": i.indicator_value}
            for i in indicators[-12:]  # 最近 12 期
        ]

        return {
            "indicator": indicator_name,
            "frequency": indicators[0].frequency if indicators else "",
            "count": len(data),
            "latest": data[-1] if data else None,
            "data": data,
        }

    # ----- 最新行情概览 -----

    @registry.register(
        name="get_market_overview",
        description=(
            "获取某市场所有已同步股票的最新行情概览。"
            "适用于: 市场整体表现分析、涨跌榜。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "market": {
                    "type": "string",
                    "description": "市场: US (美股) 或 CN (A股)",
                },
            },
            "required": ["market"],
        },
        role=ToolRole.DATA_QUERY,
    )
    def get_market_overview(market: str = "CN") -> dict:
        """获取市场概览."""
        if not data_service:
            return {"error": "DataService 未初始化"}

        df = data_service.get_latest_prices(market.upper())
        if df.empty:
            return {"market": market, "count": 0, "message": "暂无数据"}

        records = []
        for _, row in df.head(20).iterrows():
            records.append({
                "symbol": row.get("symbol", ""),
                "name": row.get("name", ""),
                "close": row.get("close"),
                "change_pct": row.get("change_pct"),
                "volume": row.get("volume"),
            })

        return {
            "market": market.upper(),
            "count": len(df),
            "stocks": records,
        }

    logger.info(
        "Registered %d data tools: %s",
        4, ["get_stock_price", "search_stocks", "get_macro_indicator", "get_market_overview"],
    )


def register_sql_tool(registry: ToolRegistry, db_path: str = "data/finance.db") -> None:
    """将 Text2SQL 引擎注册为 MCP 工具 (execute_sql).

    直接对本地 SQLite 数据库执行 SELECT 查询, 返回结构化结果。
    对应 Agent 五大工具之: 数据库查询工具 (ToolRole.TEXT2SQL)。

    安全限制:
        - 仅允许 SELECT / WITH 开头的只读查询
        - 结果最多返回 100 行
        - 查询超时由 SQLite busy timeout 兜底
    """
    import sqlite3

    def _execute_sql(query: str) -> dict:
        """执行只读 SQL 查询."""
        stripped = query.strip()
        if not stripped:
            return {"error": "空查询", "rows": [], "row_count": 0}

        # 只读白名单: 仅 SELECT / WITH
        head = stripped.upper()
        if not (head.startswith("SELECT") or head.startswith("WITH")):
            return {
                "error": "仅允许只读 SELECT 查询 (禁止 INSERT/UPDATE/DELETE/DDL)",
                "rows": [], "row_count": 0,
            }

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(stripped + " LIMIT 100")
            rows = [dict(r) for r in cur.fetchmany(100)]
            columns = [d[0] for d in cur.description] if cur.description else []
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }
        except Exception as e:
            return {"error": f"SQL 执行失败: {e}", "rows": [], "row_count": 0}
        finally:
            conn.close()

    @registry.register(
        name="execute_sql",
        description=(
            "对本地金融数据库执行 SQL 查询 (只读 SELECT)。"
            "数据库含 stocks 表 (股票代码/名称/行业) 和 daily_prices 表 "
            "(日线行情: 开高低收/成交量/涨跌幅/换手率, 前复权, 2023-08 至今, 10 只 A 股)。"
            "适用: 统计类问题 (最高/最低/平均值/排名/天数统计)、跨股票对比。"
            "示例: SELECT s.symbol, s.name, dp.trade_date, dp.close FROM daily_prices dp "
            "JOIN stocks s ON dp.stock_id=s.id WHERE s.symbol='600519' "
            "AND dp.trade_date BETWEEN '2024-01-01' AND '2024-12-31' ORDER BY dp.trade_date;"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "只读 SQL 查询语句 (仅 SELECT/WITH)",
                },
            },
            "required": ["query"],
        },
        role=ToolRole.TEXT2SQL,
    )
    def execute_sql(query: str) -> dict:
        return _execute_sql(query)

    logger.info("Registered sql tool: execute_sql (Text2SQL)")
