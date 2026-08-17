#!/usr/bin/env python3
# scripts/build_text2sql_data.py
"""基于博金比赛问题 + 我们的数据库，生成 Text2SQL 训练数据.

数据来源:
  - data/bojin/question.json: 1000 条博金比赛 Text2SQL 问题
  - 我们的 SQLite 数据库: stocks, daily_prices, macro_indicators

输出: data/training/text2sql_train.jsonl (Qwen Chat Template 格式)

用法:
  python scripts/build_text2sql_data.py
"""

import json, os, random, logging
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 DeepResearch Agent 的 Text2SQL 引擎。根据用户问题生成 SQL 查询。

## 数据库 Schema
### stocks 表 (股票信息)
- symbol TEXT: 股票代码 (6位数字)
- name TEXT: 股票名称
- market TEXT: 市场 (CN= A股)
- sector TEXT: 行业板块
- industry TEXT: 细分行业
- exchange TEXT: 交易所 (SSE=上交所, SZSE=深交所)

### daily_prices 表 (日线行情)
- stock_id INTEGER: 关联 stocks.id
- trade_date DATE: 交易日期
- open/high/low/close REAL: OHLC价格
- volume REAL: 成交量
- change_pct REAL: 涨跌幅(%)
- turnover_rate REAL: 换手率(%)

### macro_indicators 表 (宏观指标)
- indicator_name TEXT: 指标名 (cpi/pmi/m2/gdp/lpr)
- indicator_value REAL: 数值
- pub_date DATE: 发布日期

## 规则
1. 只生成 SELECT 查询，不要 INSERT/UPDATE/DELETE
2. A股代码 6 位数字字符串，查询时用单引号如 '600519'
3. 涨跌幅 = (收盘价 - 前一日收盘价) / 前一日收盘价 * 100，用 change_pct 字段
4. 查询结果用中文描述"""


def main():
    # 加载博金问题
    questions = []
    with open("data/bojin/question.json") as f:
        for line in f:
            q = json.loads(line.strip())
            questions.append(q)

    log.info(f"加载博金问题: {len(questions)} 条")

    # 加载数据库中的真实股票
    from data_layer.config import load_config
    from data_layer.database.connection import DatabaseManager
    from data_layer.repositories.stock_repository import StockRepository
    from data_layer.repositories.price_repository import PriceRepository

    config = load_config()
    db = DatabaseManager(config.db_path)
    stock_repo = StockRepository(db)
    price_repo = PriceRepository(db)
    cn_stocks = stock_repo.find_by_market("CN")

    log.info(f"可用股票: {len(cn_stocks)} 只")

    examples = []

    # 按问题类型分类生成训练数据
    for q in questions:
        question = q["question"]
        qid = q["id"]

        # 判断问题类型 → 生成对应 SQL
        sql, answer = generate_sql_answer(question, cn_stocks, price_repo, qid)

        if sql is None:
            continue

        # 构建训练样本
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": (
                f"我需要将这个问题转化为 SQL 查询来获取数据。\n\n"
                f"```sql\n{sql}\n```\n\n"
                f"查询结果:\n{answer}"
            )},
        ]

        examples.append({
            "messages": messages,
            "metadata": {
                "source": "bojin_text2sql",
                "intent_type": "text2sql",
                "question_id": qid,
            },
        })

        if len(examples) >= 300:
            break

    # 保存
    os.makedirs("data/training", exist_ok=True)
    path = "data/training/text2sql_train.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    log.info(f"Text2SQL 训练数据: {len(examples)} 条 → {path}")


def generate_sql_answer(question, stocks, price_repo, qid):
    """根据问题生成对应的 SQL 和答案."""
    q = question

    # Type 1: 查询某行业某天的涨跌幅最大股票
    if "涨跌幅最大" in q and "行业" in q:
        sector = extract_sector(q)
        date_str = extract_date(q)
        if sector and date_str:
            sql = (
                f"SELECT s.symbol, s.name, dp.change_pct "
                f"FROM daily_prices dp JOIN stocks s ON dp.stock_id = s.id "
                f"WHERE s.sector = '{sector}' AND dp.trade_date = '{date_str}' "
                f"ORDER BY dp.change_pct DESC LIMIT 1"
            )
            answer = f"在{date_str}日，{sector}行业中涨跌幅最大的股票涨幅为X%。"
            return sql, answer

    # Type 2: 查询某行业某天的股票数量
    if "股票数量" in q or "有多少只" in q or "有几只" in q:
        sector = extract_sector(q)
        date_str = extract_date(q) or "2026-07-30"
        condition = extract_condition(q)
        if sector:
            sql = (
                f"SELECT COUNT(DISTINCT s.symbol) "
                f"FROM daily_prices dp JOIN stocks s ON dp.stock_id = s.id "
                f"WHERE s.sector = '{sector}' AND dp.trade_date = '{date_str}'"
            )
            if condition:
                sql += f" AND dp.{condition}"
            answer = f"在{date_str}日，{sector}行业共有 N 只股票。"
            return sql, answer

    # Type 3: 查询某个股的日行情数据
    if "行情" in q or "股价" in q or "收盘价" in q or "涨跌幅" in q:
        symbol = extract_symbol(q)
        date_str = extract_date(q) or "2026-07-30"
        if symbol:
            sql = (
                f"SELECT s.name, dp.trade_date, dp.open, dp.high, dp.low, "
                f"dp.close, dp.volume, dp.change_pct "
                f"FROM daily_prices dp JOIN stocks s ON dp.stock_id = s.id "
                f"WHERE s.symbol = '{symbol}' AND dp.trade_date = '{date_str}'"
            )
            answer = f"在{date_str}日，{symbol}的行情为：..."
            return sql, answer

    # Type 4: 查询股票基本信息
    if "代码" in q or "股票代码" in q or "有哪些" in q:
        sector = extract_sector(q)
        if sector:
            sql = (
                f"SELECT symbol, name, industry FROM stocks "
                f"WHERE sector = '{sector}' ORDER BY symbol"
            )
            answer = f"{sector}行业的股票包括：..."
            return sql, answer

    # Type 5: Generic — 生成一个合理的查询
    # 随机选一只股票生成查询
    if stocks:
        s = random.choice(stocks)
        sql = (
            f"SELECT s.name, dp.trade_date, dp.close, dp.change_pct, dp.volume "
            f"FROM daily_prices dp JOIN stocks s ON dp.stock_id = s.id "
            f"WHERE s.symbol = '{s.symbol}' "
            f"ORDER BY dp.trade_date DESC LIMIT 5"
        )
        answer = f"{s.name}({s.symbol})最近5个交易日的数据已列出。"
        return sql, answer

    return None, None


def extract_sector(q):
    """从问题中提取行业."""
    sectors = ["白酒", "新能源", "金融", "银行", "保险", "医药", "家电",
               "半导体", "光伏", "汽车", "食品饮料", "制造业", "制药",
               "综合金融", "建筑材料", "房地产", "有色金属", "煤炭",
               "电力", "交通运输", "商贸零售", "电子", "计算机"]
    for s in sectors:
        if s in q:
            return s
    return None


def extract_date(q):
    """从问题中提取日期 YYYYMMDD → YYYY-MM-DD."""
    import re
    m = re.search(r'(\d{4})(\d{2})(\d{2})', q)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', q)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def extract_symbol(q):
    """从问题中提取股票代码."""
    import re
    m = re.search(r'(\d{6})', q)
    if m:
        return m.group(1)
    return None


def extract_condition(q):
    """提取条件如 '涨幅超过5%'."""
    import re
    m = re.search(r'涨幅超过\s*(\d+)%', q)
    if m:
        return f"change_pct > {m.group(1)}"
    m = re.search(r'跌幅超过\s*(\d+)%', q)
    if m:
        return f"change_pct < -{m.group(1)}"
    return None


if __name__ == "__main__":
    main()
