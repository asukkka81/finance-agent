#!/usr/bin/env python3
"""重建 Text2SQL 训练数据 — 基于本地库真实 schema + 真实执行验证.

与旧版 (build_text2sql_data.py) 的区别:
    1. 问题从本地库的实际表结构/数据范围生成 (不再用博金问题 — 那些问题引用的
       表与日期在本地库中不存在)
    2. 每条样本的 SQL 都真实执行, 答案来自执行结果 (不再用占位模板)
    3. 只覆盖有数据的表 (macro/funds 为空则不入 schema)

输出: data/training/text2sql_train.jsonl
格式: {messages: [system(schema), user(问题), assistant(SQL+真实结果)]}
"""

import json
import logging
import random
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = "data/finance.db"
OUT_PATH = "data/training/text2sql_train.jsonl"

SYSTEM_PROMPT_TEMPLATE = """你是 DeepResearch Agent 的 Text2SQL 引擎。根据用户问题生成 SQL 查询。

## 数据库 Schema
### stocks 表 (股票信息)
- symbol TEXT: 股票代码 (6位数字)
- name TEXT: 股票名称
- market TEXT: 市场 (CN= A股)
- sector TEXT: 行业板块
- industry TEXT: 细分行业
- exchange TEXT: 交易所 (SSE=上交所, SZSE=深交所)

### daily_prices 表 (日线行情, 前复权)
- stock_id INTEGER: 关联 stocks.id
- trade_date DATE: 交易日期
- open/high/low/close REAL: OHLC价格
- volume REAL: 成交量 (股)
- adj_close REAL: 前复权收盘价
- pre_close REAL: 前一日收盘价
- change_pct REAL: 涨跌幅(%)
- turnover_rate REAL: 换手率(%)

## 规则
1. 只生成 SELECT 查询，不要 INSERT/UPDATE/DELETE
2. A股代码 6 位数字字符串，查询时用单引号如 '600519'
3. 查询结果用中文描述"""


def _fmt(v, nd=2):
    """数字格式化."""
    if v is None:
        return "N/A"
    return f"{float(v):.{nd}f}"


class Text2SQLBuilder:
    """从本地库生成可验证的 text2sql 训练样本."""

    def __init__(self, db_path: str, seed: int = 42):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.rng = random.Random(seed)
        self.samples: list[dict] = []
        self._sqlite_has_sqrt = sqlite3.sqlite_version_info >= (3, 35, 0)

        self.stocks = [
            dict(r) for r in self.conn.execute(
                "SELECT id, symbol, name, sector, industry FROM stocks ORDER BY symbol"
            )
        ]
        # 只保留有行情的股票
        self.stocks = [s for s in self.stocks if self._count(
            "SELECT COUNT(*) FROM daily_prices WHERE stock_id = ?", (s["id"],)
        ) > 0]
        logger.info("有行情数据的股票: %d 只", len(self.stocks))

        # 各股票的完整年度集合 (数据范围内)
        rows = self.conn.execute(
            "SELECT MIN(trade_date) mn, MAX(trade_date) mx FROM daily_prices"
        ).fetchone()
        self.data_min, self.data_max = rows["mn"], rows["mx"]
        self.full_years = [
            y for y in range(int(self.data_min[:4]) + 1, int(self.data_max[:4]))
        ]
        logger.info("数据范围 %s ~ %s, 完整年度: %s",
                    self.data_min, self.data_max, self.full_years)

    # ----- 工具 -----

    def _count(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()[0]

    def _exec(self, sql: str):
        """执行 SQL, 返回 (rows, error)."""
        try:
            rows = [dict(r) for r in self.conn.execute(sql)]
            return rows, None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    def _pick_stock(self):
        return self.rng.choice(self.stocks)

    def _pick_date(self, symbol: str, year: int | None = None) -> str:
        """从真实数据中随机选一个交易日."""
        sql = ("SELECT trade_date FROM daily_prices dp JOIN stocks s ON s.id=dp.stock_id "
               "WHERE s.symbol = ?")
        params: list = [symbol]
        if year:
            sql += " AND dp.trade_date BETWEEN ? AND ?"
            params += [f"{year}-01-01", f"{year}-12-31"]
        sql += " ORDER BY dp.trade_date"
        dates = [r[0] for r in self.conn.execute(sql, params)]
        return self.rng.choice(dates)

    def _add(self, question: str, sql: str, answer_text: str) -> bool:
        """执行验证后加入样本. 返回是否成功."""
        rows, err = self._exec(sql)
        if err is not None or not rows:
            logger.warning("SQL 验证失败 (丢弃): %s | %s | %s", question, sql, err)
            return False
        if "N/A" in answer_text:
            logger.warning("答案含 N/A (丢弃): %s", question)
            return False
        self.samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {"role": "user", "content": question},
                {"role": "assistant", "content": (
                    "我需要将这个问题转化为 SQL 查询来获取数据。\n\n"
                    f"```sql\n{sql}\n```\n\n查询结果:\n{answer_text}"
                )},
            ],
            "metadata": {"source": "rebuild_text2sql", "validated": True},
        })
        return True

    # ----- 问题模板 (全部基于真实数据可验证) -----

    def gen_interval_return(self, n: int = 40):
        """T1: 区间涨跌幅."""
        for _ in range(n):
            s = self._pick_stock()
            d1 = self._pick_date(s["symbol"])
            d2 = self._pick_date(s["symbol"])
            if d1 > d2:
                d1, d2 = d2, d1
            q = f"{s['name']}({s['symbol']}) 从 {d1} 到 {d2} 的涨跌幅是多少？百分数保留两位小数。"
            sql = (
                "WITH bounds AS ("
                "  SELECT "
                "    (SELECT close FROM daily_prices dp JOIN stocks st ON st.id=dp.stock_id "
                f"     WHERE st.symbol = '{s['symbol']}' AND dp.trade_date >= '{d1}' "
                "     ORDER BY dp.trade_date LIMIT 1) AS first_close,"
                "    (SELECT close FROM daily_prices dp JOIN stocks st ON st.id=dp.stock_id "
                f"     WHERE st.symbol = '{s['symbol']}' AND dp.trade_date <= '{d2}' "
                "     ORDER BY dp.trade_date DESC LIMIT 1) AS last_close"
                ") "
                "SELECT ROUND((last_close / first_close - 1) * 100, 2) AS ret FROM bounds"
            )
            rows, err = self._exec(sql)
            if err or not rows:
                continue
            ans = f"{s['name']}({s['symbol']}) 在 {d1} 至 {d2} 区间涨跌幅为 {_fmt(rows[0]['ret'])}%。"
            self._add(q, sql, ans)

    def gen_high_low(self, n: int = 40):
        """T2/T3: 区间最高/最低收盘价及日期."""
        for _ in range(n):
            s = self._pick_stock()
            year = self.rng.choice(self.full_years)
            for extreme, order, desc in (
                ("最高", "DESC", "高"), ("最低", "ASC", "低"),
            ):
                q = (f"{s['name']}({s['symbol']}) 在 {year} 年收盘价{extreme}的交易日是哪天？"
                     f"收盘价是多少？")
                sql = (
                    "SELECT dp.trade_date, dp.close FROM daily_prices dp "
                    f"JOIN stocks st ON st.id = dp.stock_id WHERE st.symbol = '{s['symbol']}' "
                    f"AND dp.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31' "
                    f"ORDER BY dp.close {order} LIMIT 1"
                )
                rows, err = self._exec(sql)
                if err or not rows:
                    continue
                ans = (f"{s['name']}({s['symbol']}) 在 {year} 年收盘价最{desc}的交易日是 "
                       f"{rows[0]['trade_date']}，收盘价 {_fmt(rows[0]['close'])} 元。")
                self._add(q, sql, ans)

    def gen_single_day(self, n: int = 50):
        """T4: 单日行情."""
        for _ in range(n):
            s = self._pick_stock()
            d = self._pick_date(s["symbol"])
            q = (f"{s['name']}({s['symbol']}) 在 {d} 的开盘价、收盘价和涨跌幅分别是多少？")
            sql = (
                "SELECT dp.open, dp.close, dp.change_pct FROM daily_prices dp "
                f"JOIN stocks st ON st.id = dp.stock_id WHERE st.symbol = '{s['symbol']}' "
                f"AND dp.trade_date = '{d}'"
            )
            rows, err = self._exec(sql)
            if err or not rows:
                continue
            r = rows[0]
            ans = (f"{s['name']}({s['symbol']}) 在 {d} 开盘价 {_fmt(r['open'])} 元，"
                   f"收盘价 {_fmt(r['close'])} 元，涨跌幅 {_fmt(r['change_pct'])}%。")
            self._add(q, sql, ans)

    def gen_max_volume(self, n: int = 30):
        """T5: 年度成交量最大日."""
        for _ in range(n):
            s = self._pick_stock()
            year = self.rng.choice(self.full_years)
            q = f"{s['name']}({s['symbol']}) 在 {year} 年成交量最大的交易日是哪天？成交量是多少？"
            sql = (
                "SELECT dp.trade_date, dp.volume FROM daily_prices dp "
                f"JOIN stocks st ON st.id = dp.stock_id WHERE st.symbol = '{s['symbol']}' "
                f"AND dp.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31' "
                "ORDER BY dp.volume DESC LIMIT 1"
            )
            rows, err = self._exec(sql)
            if err or not rows:
                continue
            ans = (f"{s['name']}({s['symbol']}) 在 {year} 年成交量最大的交易日是 "
                   f"{rows[0]['trade_date']}，成交量 {_fmt(rows[0]['volume'], 0)} 股。")
            self._add(q, sql, ans)

    def gen_year_rank(self):
        """T6: 年度涨幅排行 (全池)."""
        for year in self.full_years:
            q = f"在 {year} 年，数据库中 A 股涨幅最高的是哪只股票？涨幅是多少？百分数保留两位小数。"
            sql = (
                "WITH ranges AS ("
                f"  SELECT stock_id, MIN(trade_date) AS sd, MAX(trade_date) AS ed "
                f"  FROM daily_prices WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31' "
                "  GROUP BY stock_id"
                "), returns AS ("
                "  SELECT r.stock_id, (e.close / f.close - 1) * 100 AS ret"
                "  FROM ranges r"
                "  JOIN daily_prices f ON f.stock_id = r.stock_id AND f.trade_date = r.sd"
                "  JOIN daily_prices e ON e.stock_id = r.stock_id AND e.trade_date = r.ed"
                ")"
                "SELECT s.symbol, s.name, ROUND(r.ret, 2) AS ret FROM returns r "
                "JOIN stocks s ON s.id = r.stock_id ORDER BY r.ret DESC LIMIT 1"
            )
            rows, err = self._exec(sql)
            if err or not rows:
                continue
            ans = (f"{year} 年涨幅最高的是 {rows[0]['name']}({rows[0]['symbol']})，"
                   f"涨幅 {_fmt(rows[0]['ret'])}%。")
            self._add(q, sql, ans)

    def gen_moving_avg(self, n: int = 40, window: int = 20):
        """T7: N 日均线 (窗口函数)."""
        for _ in range(n):
            s = self._pick_stock()
            d = self._pick_date(s["symbol"])
            # 确保日期之前有足够历史 (先取该日及之前第 window 个交易日)
            sql_check = (
                "SELECT COUNT(*) FROM ("
                "  SELECT dp.trade_date FROM daily_prices dp "
                f"  JOIN stocks st ON st.id = dp.stock_id WHERE st.symbol = '{s['symbol']}' "
                f"  AND dp.trade_date <= '{d}' ORDER BY dp.trade_date DESC LIMIT ?)"
            )
            if self._count(sql_check, (window,)) < window:
                continue
            q = f"{s['name']}({s['symbol']}) 在 {d} 的 {window} 日移动平均收盘价是多少？"
            sql = (
                f"SELECT ROUND(AVG(close) OVER (ORDER BY trade_date "
                f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW), 2) AS ma FROM daily_prices dp "
                f"JOIN stocks st ON st.id = dp.stock_id WHERE st.symbol = '{s['symbol']}' "
                f"AND dp.trade_date <= '{d}' ORDER BY dp.trade_date DESC LIMIT 1"
            )
            rows, err = self._exec(sql)
            if err or not rows or rows[0]["ma"] is None:
                continue
            ans = (f"{s['name']}({s['symbol']}) 在 {d} 的 {window} 日均线收盘价为 "
                   f"{_fmt(rows[0]['ma'])} 元。")
            self._add(q, sql, ans)

    def gen_volatility(self, n: int = 30):
        """T8: 年度涨跌幅波动 (标准差)."""
        for _ in range(n):
            s = self._pick_stock()
            year = self.rng.choice(self.full_years)
            q = f"{s['name']}({s['symbol']}) 在 {year} 年日涨跌幅的标准差是多少？保留四位小数。"
            if self._sqlite_has_sqrt:
                sql = (
                    f"WITH stats AS (SELECT AVG(change_pct) AS m FROM daily_prices dp "
                    f"JOIN stocks st ON st.id = dp.stock_id WHERE st.symbol = '{s['symbol']}' "
                    f"AND dp.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31') "
                    "SELECT ROUND(SQRT(AVG((change_pct - m) * (change_pct - m))), 4) AS v "
                    "FROM daily_prices, stats "
                    f"WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'"
                )
            else:
                sql = (
                    f"WITH stats AS (SELECT AVG(change_pct) AS m FROM daily_prices dp "
                    f"JOIN stocks st ON st.id = dp.stock_id WHERE st.symbol = '{s['symbol']}' "
                    f"AND dp.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31') "
                    "SELECT ROUND(AVG(ABS(change_pct - m)), 4) AS v FROM daily_prices, stats "
                    f"WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'"
                )
            rows, err = self._exec(sql)
            if err or not rows or rows[0]["v"] is None:
                continue
            metric = "标准差" if self._sqlite_has_sqrt else "平均绝对偏离"
            ans = (f"{s['name']}({s['symbol']}) 在 {year} 年日涨跌幅的{metric}为 "
                   f"{_fmt(rows[0]['v'], 4)}。")
            self._add(q, sql, ans)

    def gen_avg_close(self, n: int = 30):
        """T9: 年度平均收盘价."""
        for _ in range(n):
            s = self._pick_stock()
            year = self.rng.choice(self.full_years)
            q = f"{s['name']}({s['symbol']}) 在 {year} 年的平均收盘价是多少？保留两位小数。"
            sql = (
                "SELECT ROUND(AVG(dp.close), 2) AS avg_close FROM daily_prices dp "
                f"JOIN stocks st ON st.id = dp.stock_id WHERE st.symbol = '{s['symbol']}' "
                f"AND dp.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'"
            )
            rows, err = self._exec(sql)
            if err or not rows or rows[0]["avg_close"] is None:
                continue
            ans = f"{s['name']}({s['symbol']}) 在 {year} 年平均收盘价为 {_fmt(rows[0]['avg_close'])} 元。"
            self._add(q, sql, ans)

    def gen_up_days(self, n: int = 30):
        """T10: 年度上涨天数."""
        for _ in range(n):
            s = self._pick_stock()
            year = self.rng.choice(self.full_years)
            q = f"{s['name']}({s['symbol']}) 在 {year} 年上涨（涨跌幅大于0）的天数有多少天？"
            sql = (
                "SELECT COUNT(*) FROM daily_prices dp "
                f"JOIN stocks st ON st.id = dp.stock_id WHERE st.symbol = '{s['symbol']}' "
                f"AND dp.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31' "
                "AND dp.change_pct > 0"
            )
            rows, err = self._exec(sql)
            if err or not rows:
                continue
            ans = f"{s['name']}({s['symbol']}) 在 {year} 年上涨天数为 {next(iter(rows[0].values()))} 天。"
            self._add(q, sql, ans)

    def gen_recent_list(self, n: int = 40, days: int = 5):
        """T11: 最近 N 个交易日行情."""
        for _ in range(n):
            s = self._pick_stock()
            q = f"请列出 {s['name']}({s['symbol']}) 最近 {days} 个交易日的收盘价和涨跌幅。"
            sql = (
                "SELECT dp.trade_date, dp.close, dp.change_pct FROM daily_prices dp "
                f"JOIN stocks st ON st.id = dp.stock_id WHERE st.symbol = '{s['symbol']}' "
                f"ORDER BY dp.trade_date DESC LIMIT {days}"
            )
            rows, err = self._exec(sql)
            if err or not rows or len(rows) < days:
                continue
            detail = "；".join(
                f"{r['trade_date']} 收盘 {_fmt(r['close'])} 元，涨跌 {_fmt(r['change_pct'])}%"
                for r in rows
            )
            ans = f"{s['name']}({s['symbol']}) 最近 {days} 个交易日：{detail}。"
            self._add(q, sql, ans)

    def gen_pair_compare(self, n: int = 30):
        """T12: 双股年度涨幅对比."""
        pairs = [(self.stocks[i], self.stocks[j])
                 for i in range(len(self.stocks)) for j in range(i + 1, len(self.stocks))]
        for _ in range(n):
            a, b = self.rng.choice(pairs)
            year = self.rng.choice(self.full_years)
            q = (f"在 {year} 年，{a['name']}({a['symbol']}) 和 {b['name']}({b['symbol']}) "
                 "谁的涨幅更高？分别涨了多少？百分数保留两位小数。")
            sql = (
                "WITH ranges AS ("
                f"  SELECT stock_id, MIN(trade_date) AS sd, MAX(trade_date) AS ed "
                f"  FROM daily_prices WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31' "
                "  GROUP BY stock_id"
                "), returns AS ("
                "  SELECT r.stock_id, (e.close / f.close - 1) * 100 AS ret"
                "  FROM ranges r"
                "  JOIN daily_prices f ON f.stock_id = r.stock_id AND f.trade_date = r.sd"
                "  JOIN daily_prices e ON e.stock_id = r.stock_id AND e.trade_date = r.ed"
                ")"
                "SELECT s.symbol, s.name, ROUND(r.ret, 2) AS ret FROM returns r "
                f"JOIN stocks s ON s.id = r.stock_id "
                f"WHERE s.symbol IN ('{a['symbol']}', '{b['symbol']}') ORDER BY r.ret DESC"
            )
            rows, err = self._exec(sql)
            if err or not rows or len(rows) < 2:
                continue
            high, low = rows[0], rows[1]
            ans = (f"{year} 年 {high['name']}({high['symbol']}) 涨幅更高，为 {_fmt(high['ret'])}%；"
                   f"{low['name']}({low['symbol']}) 涨幅为 {_fmt(low['ret'])}%。")
            self._add(q, sql, ans)

    # ----- 主流程 -----

    def build(self) -> list[dict]:
        logger.info("开始生成样本...")
        self.gen_interval_return(40)
        self.gen_high_low(40)
        self.gen_single_day(50)
        self.gen_max_volume(30)
        self.gen_year_rank()
        self.gen_moving_avg(40)
        self.gen_volatility(30)
        self.gen_avg_close(30)
        self.gen_up_days(30)
        self.gen_recent_list(40)
        self.gen_pair_compare(30)
        logger.info("生成完成: %d 条 (全部经真实执行验证)", len(self.samples))
        return self.samples

    def save(self, out_path: str) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info("已保存 %d 条到 %s", len(self.samples), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    builder = Text2SQLBuilder(DB_PATH)
    builder.build()
    builder.save(OUT_PATH)

    # 复验: 所有 SQL 可执行且非空
    bad = 0
    conn = sqlite3.connect(DB_PATH)
    for line in Path(OUT_PATH).open():
        d = json.loads(line)
        content = d["messages"][-1]["content"]
        sql_block = content.split("```sql\n")[1].split("\n```")[0]
        try:
            rows = conn.execute(sql_block).fetchall()
            if not rows:
                bad += 1
        except Exception as e:  # noqa: BLE001
            logger.error("复验失败: %s", e)
            bad += 1
    print(f"复验结果: {bad} 条异常 / 共 {len(builder.samples)} 条")


if __name__ == "__main__":
    main()
