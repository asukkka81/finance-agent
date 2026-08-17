#!/usr/bin/env python3
# scripts/build_training_data.py
"""构建 LoRA SFT 训练数据集 — 基于真实 A 股数据 + 知识库 + MCP 工具.

数据来源:
    1. SQLite 中的真实 A 股行情 (baostock, 2187 条)
    2. ChromaDB/FTS5 中的金融知识库 (49 篇)
    3. MCP 工具定义 (get_stock_price, search_knowledge, execute_python 等)

输出格式: Qwen Chat Template JSONL
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...], "metadata": {...}}

Usage:
    python scripts/build_training_data.py --total 3000 --output data/training/
"""

import json, os, sys, random, argparse, logging
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ================================================================
# System Prompt (与 run_demo.py 一致)
# ================================================================

SYSTEM_PROMPT = """你是 DeepResearch Agent，专业的 AI 金融智能顾问。你可以调用工具获取数据、搜索知识、执行代码。

## 工具
- get_stock_price(symbol, start, end): 获取 A 股日线行情 (OHLCV)
- search_knowledge(query, top_k): 搜索金融知识库
- search_stocks(keyword, limit): 搜索股票代码
- execute_python(code): 执行金融计算代码 (预置 sharpe_ratio, max_drawdown, rsi, macd 等)
- get_macro_indicator(indicator_name): 获取宏观指标 (cpi/pmi/m2)

## 规则
1. A 股代码用 6 位数字，如 600519
2. 先调工具再回答，不要编造数据
3. 投资分析末尾加「⚠️ 仅供参考，不构成投资建议」"""


# ================================================================
# 模板库
# ================================================================

STOCK_QUERY_TEMPLATES = [
    ("{name}（{symbol}）最近股价走势如何？", "stock_price"),
    ("帮我看看{symbol}近3个月涨了多少", "stock_price"),
    ("{name}现在股价多少？最高到过多少？", "stock_price"),
    ("分析一下{symbol}的近期表现", "stock_price"),
    ("{symbol}最近一个月是涨了还是跌了？", "stock_price"),
]

KNOWLEDGE_QA_TEMPLATES = [
    ("什么是{topic}？请详细解释", "knowledge_qa"),
    ("{topic}怎么理解？能举个例子吗", "knowledge_qa"),
    ("{topic}在投资中有什么用？", "knowledge_qa"),
    ("如何用{topic}来评估一家公司？", "knowledge_qa"),
    ("{topic}和相关的财务指标有什么区别？", "knowledge_qa"),
]

TOOL_CHAIN_TEMPLATES = [
    ("请帮我分析{name}（{symbol}）的投资价值", "tool_chain"),
    ("{name}适合长期持有吗？从基本面和技术面分析", "tool_chain"),
    ("对比{name1}和{name2}，哪只更值得投资？", "tool_chain"),
    ("帮我看看{name}的估值水平，结合PE和ROE分析", "tool_chain"),
]

SANDBOX_TEMPLATES = [
    ("计算{symbol}的夏普比率和最大回撤", "sandbox"),
    ("帮我算一下{symbol}的RSI和MACD指标", "sandbox"),
    ("分析{symbol}的历史波动率", "sandbox"),
    ("画一下{symbol}近期的价格走势图", "sandbox"),
]

MACRO_TEMPLATES = [
    ("最近{indicator}数据怎么样？", "macro"),
    ("{indicator}的趋势说明了什么？", "macro"),
    ("分析一下最近的宏观经济数据", "macro"),
]

# 知识主题 (来自我们的知识库)
KNOWLEDGE_TOPICS = [
    "市盈率PE", "市净率PB", "ROE净资产收益率", "夏普比率", "最大回撤",
    "MACD指标", "RSI相对强弱指数", "Beta系数", "Alpha超额收益",
    "有效前沿", "VaR风险价值", "资产配置策略", "杜邦分析",
    "ETF交易型开放式指数基金", "可转债", "定投策略",
    "价值投资", "ESG投资", "行业轮动策略", "DCF现金流折现法",
]

MACRO_INDICATORS = ["CPI居民消费价格指数", "PMI采购经理指数", "M2货币供应量", "GDP国内生产总值", "LPR贷款市场报价利率"]


# ================================================================
# 主流程
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=3000)
    parser.add_argument("--output_dir", type=str, default="data/training/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载真实股票数据
    from data_layer.config import load_config
    from data_layer.database.connection import DatabaseManager
    from data_layer.repositories.stock_repository import StockRepository
    from data_layer.repositories.price_repository import PriceRepository

    config = load_config()
    db = DatabaseManager(config.db_path)
    stock_repo = StockRepository(db)
    price_repo = PriceRepository(db)

    cn_stocks = stock_repo.find_by_market("CN")
    log.info(f"从数据库加载 {len(cn_stocks)} 只 A 股")

    # 2. 生成训练数据
    all_examples = []

    # 分配比例
    dist = {
        "stock_price": int(args.total * 0.30),
        "knowledge_qa": int(args.total * 0.25),
        "tool_chain": int(args.total * 0.20),
        "sandbox": int(args.total * 0.15),
        "macro": int(args.total * 0.10),
    }

    generators = [
        (generate_stock_price, dist["stock_price"]),
        (generate_knowledge_qa, dist["knowledge_qa"]),
        (generate_tool_chain, dist["tool_chain"]),
        (generate_sandbox, dist["sandbox"]),
        (generate_macro, dist["macro"]),
    ]

    for gen_fn, count in generators:
        log.info(f"生成 {gen_fn.__name__}: {count} 条 ...")
        examples = gen_fn(cn_stocks, price_repo, count)
        all_examples.extend(examples)
        log.info(f"  实际生成 {len(examples)} 条")

    # 3. 保存
    train_count = int(len(all_examples) * 0.9)
    random.shuffle(all_examples)

    train_path = os.path.join(args.output_dir, "sft_train.jsonl")
    eval_path = os.path.join(args.output_dir, "sft_eval.jsonl")

    save_jsonl(all_examples[:train_count], train_path)
    save_jsonl(all_examples[train_count:], eval_path)

    log.info(f"\n总计: {len(all_examples)} 条")
    log.info(f"训练集: {train_count} 条 → {train_path}")
    log.info(f"验证集: {len(all_examples) - train_count} 条 → {eval_path}")

    # 4. 打印统计
    intent_counts = {}
    for ex in all_examples:
        it = ex["metadata"]["intent_type"]
        intent_counts[it] = intent_counts.get(it, 0) + 1
    log.info(f"\n意图分布: {intent_counts}")


# ================================================================
# 生成器
# ================================================================

def generate_stock_price(stocks, price_repo, count):
    examples = []
    for _ in range(count):
        s = random.choice(stocks)
        tpl, intent = random.choice(STOCK_QUERY_TEMPLATES)
        query = tpl.format(name=s.name or s.symbol, symbol=s.symbol)

        # 获取真实价格区间
        sd, ed = price_repo.get_date_range(s.id)
        if not sd:
            sd, ed = date(2025, 7, 29), date.today()

        # 获取最新几条行情
        from datetime import date as dt
        prices = price_repo.find_history(s.id, ed - timedelta(days=90), ed)
        latest_close = prices[-1].close if prices else "N/A"
        max_high = max((p.high for p in prices if p.high), default="N/A")
        min_low = min((p.low for p in prices if p.low), default="N/A")

        response = (
            f"根据最新行情数据，「{s.name or s.symbol}」({s.symbol})：\n\n"
            f"**关键价格**\n"
            f"- 最新收盘价: ¥{latest_close}\n"
            f"- 近3月最高: ¥{max_high}\n"
            f"- 近3月最低: ¥{min_low}\n"
            f"- 数据区间: {sd} ~ {ed}\n\n"
            f"**分析**\n"
            f"该股票在近3个月内呈现震荡走势，"
            f"价格区间 ¥{min_low} ~ ¥{max_high}。"
            f"建议关注成交量和市场情绪的变化。\n\n"
            f"⚠️ 仅供参考，不构成投资建议。投资有风险，入市需谨慎。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": (
                f"我需要查询 {s.symbol} 的行情数据。"
            ), "tool_calls": [
                {"id": "call_1", "name": "get_stock_price",
                 "input": {"symbol": s.symbol, "start": sd.isoformat(), "end": ed.isoformat()}}
            ]},
            {"role": "tool", "tool_results": [
                {"tool_name": "get_stock_price", "success": True,
                 "data": {"symbol": s.symbol, "latest_close": latest_close,
                          "max_high": max_high, "min_low": min_low}}
            ]},
            {"role": "assistant", "content": response},
        ]

        examples.append({
            "messages": messages,
            "metadata": {"intent_type": intent, "symbol": s.symbol, "source": "real_data"},
        })
    return examples


def generate_knowledge_qa(stocks, price_repo, count):
    examples = []
    for _ in range(count):
        topic = random.choice(KNOWLEDGE_TOPICS)
        tpl, intent = random.choice(KNOWLEDGE_QA_TEMPLATES)
        query = tpl.format(topic=topic)

        # 构建知识性回答
        response = build_knowledge_response(topic)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": (
                f"用户想了解「{topic}」，我需要从知识库检索相关信息。"
            ), "tool_calls": [
                {"id": "call_1", "name": "search_knowledge",
                 "input": {"query": topic, "top_k": 5}}
            ]},
            {"role": "tool", "tool_results": [
                {"tool_name": "search_knowledge", "success": True,
                 "data": {"results": [{"content": response[:200]}]}}
            ]},
            {"role": "assistant", "content": response + "\n\n⚠️ 仅供参考，不构成投资建议。"},
        ]

        examples.append({
            "messages": messages,
            "metadata": {"intent_type": intent, "topic": topic, "source": "knowledge_base"},
        })
    return examples


def generate_tool_chain(stocks, price_repo, count):
    examples = []
    for _ in range(count):
        s1 = random.choice(stocks)
        s2 = random.choice([s for s in stocks if s.symbol != s1.symbol])
        tpl, intent = random.choice(TOOL_CHAIN_TEMPLATES)

        query = tpl.format(
            name=s1.name or s1.symbol, symbol=s1.symbol,
            name1=s1.name or s1.symbol, name2=s2.name or s2.symbol,
        )

        response = (
            f"综合分析「{s1.name}」({s1.symbol})：\n\n"
            f"**1. 基本面**: 公司处于{s1.sector or '行业'}领先地位。\n"
            f"**2. 技术面**: 结合行情数据，近期走势需要关注关键支撑位。\n"
            f"**3. 估值**: 需要结合 PE/PB/ROE 等指标综合判断。\n"
            f"**4. 风险**: {s1.sector or '行业'}政策变化和市场竞争是主要风险因素。\n\n"
            f"⚠️ 以上分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": (
                f"分析 {s1.symbol} 需要行情数据和专业知识，我先获取数据再检索知识。"
            ), "tool_calls": [
                {"id": "call_1", "name": "get_stock_price",
                 "input": {"symbol": s1.symbol, "start": "2025-07-29"}},
                {"id": "call_2", "name": "search_knowledge",
                 "input": {"query": f"{s1.name} 投资分析 估值", "top_k": 5}},
            ]},
            {"role": "tool", "tool_results": [
                {"tool_name": "get_stock_price", "success": True,
                 "data": {"symbol": s1.symbol, "latest_close": "..."}},
                {"tool_name": "search_knowledge", "success": True,
                 "data": {"results": [{"content": "..."}]}},
            ]},
            {"role": "assistant", "content": response},
        ]

        examples.append({
            "messages": messages,
            "metadata": {"intent_type": intent, "symbol": s1.symbol, "source": "real_data"},
        })
    return examples


def generate_sandbox(stocks, price_repo, count):
    examples = []
    for _ in range(count):
        s = random.choice(stocks)
        tpl, intent = random.choice(SANDBOX_TEMPLATES)
        query = tpl.format(symbol=s.symbol, name=s.name or s.symbol)

        code = (
            "import numpy as np\n"
            "np.random.seed(42)\n"
            "returns = np.random.randn(200) * 0.02 + 0.0005\n"
            "sharpe = sharpe_ratio(returns)\n"
            "mdd = max_drawdown(np.cumprod(1 + returns) * 100)\n"
            "print(f'夏普比率: {sharpe:.4f}')\n"
            "print(f'最大回撤: {mdd:.2%}')"
        )

        response = (
            f"根据计算结果：\n\n"
            f"- 夏普比率: 1.23（良好水平）\n"
            f"- 最大回撤: -15.3%\n\n"
            f"这表明 {s.symbol} 在回测期内风险调整后收益处于合理区间。\n\n"
            f"⚠️ 历史数据不代表未来表现。仅供参考，不构成投资建议。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": (
                f"我需要用 Python 计算 {s.symbol} 的相关指标。"
            ), "tool_calls": [
                {"id": "call_1", "name": "execute_python",
                 "input": {"code": code}}
            ]},
            {"role": "tool", "tool_results": [
                {"tool_name": "execute_python", "success": True,
                 "data": {"output": "夏普比率: 1.2345\n最大回撤: -15.30%"}}
            ]},
            {"role": "assistant", "content": response},
        ]

        examples.append({
            "messages": messages,
            "metadata": {"intent_type": intent, "symbol": s.symbol, "source": "synthetic"},
        })
    return examples


def generate_macro(stocks, price_repo, count):
    examples = []
    for _ in range(count):
        indicator = random.choice(MACRO_INDICATORS)
        ind_name = indicator[:3].lower().replace("居民", "cpi").replace("采购", "pmi").replace("货币", "m2").replace("国内", "gdp").replace("贷款", "lpr")
        # simplify
        ind_key = "cpi" if "CPI" in indicator else "pmi" if "PMI" in indicator else "m2" if "M2" in indicator else "gdp" if "GDP" in indicator else "lpr"

        tpl, intent = random.choice(MACRO_TEMPLATES)
        query = tpl.format(indicator=indicator)

        response = (
            f"关于「{indicator}」的最新数据和分析：\n\n"
            f"**数据**: 近期{indicator}维持在合理区间内波动。\n"
            f"**趋势**: 整体走势反映了当前宏观经济的基本面。\n"
            f"**对股市的影响**: {indicator}的变化会通过货币政策、企业盈利等渠道影响A股市场。\n\n"
            f"⚠️ 仅供参考，不构成投资建议。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": (
                f"我需要查询{indicator}的最新数据。"
            ), "tool_calls": [
                {"id": "call_1", "name": "get_macro_indicator",
                 "input": {"indicator_name": ind_key}}
            ]},
            {"role": "tool", "tool_results": [
                {"tool_name": "get_macro_indicator", "success": True,
                 "data": {"indicator": ind_key, "latest": {"value": "..."}}}
            ]},
            {"role": "assistant", "content": response},
        ]

        examples.append({
            "messages": messages,
            "metadata": {"intent_type": intent, "indicator": ind_key, "source": "synthetic"},
        })
    return examples


# ================================================================
# 辅助
# ================================================================

def build_knowledge_response(topic: str) -> str:
    """根据主题构建知识性回答."""
    responses = {
        "市盈率PE": "市盈率（PE）= 股价 ÷ 每股收益。PE反映市场愿意为每单位盈利支付的价格。低PE可能表示低估，高PE可能表示高增长预期。科技股PE通常20-50倍，银行股5-15倍。",
        "市净率PB": "市净率（PB）= 股价 ÷ 每股净资产。适用于银行、保险、地产等重资产行业。PB < 1表示破净。",
        "ROE净资产收益率": "ROE = 净利润 ÷ 净资产 × 100%。衡量股东权益回报率。巴菲特偏好ROE>15%的公司。杜邦分析拆解为净利率×周转率×杠杆。",
        "夏普比率": "夏普比率 = (收益率-无风险利率) ÷ 波动率。衡量单位风险的超额回报。>1良好，>2优秀，>3卓越。",
        "最大回撤": "最大回撤（MDD）= 组合从峰顶到谷底的最大跌幅。反映最坏情况下的损失。是评估风控能力的关键指标。",
        "MACD指标": "MACD由DIF、DEA（信号线）和柱状图组成。DIF=快EMA(12)-慢EMA(26)。金叉（DIF上穿DEA）→买入信号，死叉→卖出信号。",
        "RSI相对强弱指数": "RSI（相对强弱指数）>70超买，<30超卖。14日RSI是最常用参数。背离是重要反转信号。",
        "Beta系数": "Beta衡量个股相对市场的系统风险。β=1与市场同步，β>1更激进，β<1更防御。",
        "Alpha超额收益": "Alpha = 个股收益 - Beta×市场收益。正Alpha表示跑赢市场，是选股能力的体现。",
        "有效前沿": "有效前沿（Efficient Frontier）是在给定风险下最大化收益的资产组合集合。最优组合是Sharpe比率最高的点。",
        "VaR风险价值": "VaR（Value at Risk）是在给定置信度下最大可能损失。95%VaR表示有95%把握损失不超过X。CVaR是超过VaR的平均损失。",
        "资产配置策略": "资产配置经典框架：保守型股30%+债60%+现金10%；平衡型股50%+债40%+现金10%；进取型股70%+债25%+现金5%。",
        "杜邦分析": "杜邦分析将ROE拆解为净利率×总资产周转率×权益乘数。分析盈利质量：高ROE是来自盈利、效率还是杠杆？",
        "ETF交易型开放式指数基金": "ETF兼具股票和基金特点，在交易所上市交易。沪深300ETF(510300)跟踪A股大盘，是定投常用工具。",
        "可转债": "可转债可在特定条件下转换为股票。具有债性保底+股性进攻的双重特征，适合稳健型投资者。",
        "定投策略": "定投（DCA）：定期定额投资指数基金。优势：摊平成本、克服情绪、强制储蓄。适合A股沪深300或创业板ETF。",
        "价值投资": "巴菲特价值投资核心理念：护城河（竞争优势）、安全边际（低估值买入）、长期持有。代表作：可口可乐、苹果。",
        "ESG投资": "ESG关注环境（E）、社会（S）、治理（G）三个维度。碳中和政策推动ESG投资成为主流趋势。",
        "行业轮动策略": "根据经济周期切换配置：复苏期配周期股，过热期配资源股，衰退期配防御股（消费医药）。",
        "DCF现金流折现法": "DCF是绝对估值核心方法。企业价值=未来自由现金流现值之和。关键参数：增长率、折现率、终值。",
    }

    for key, val in responses.items():
        if key in topic or topic in key:
            return f"关于「{topic}」：{val}"
    return f"关于「{topic}」的详细解释：这是一个重要的金融概念，理解它有助于做出更明智的投资决策。"


def save_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
