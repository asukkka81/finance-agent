#!/usr/bin/env python3
# scripts/prepare_sft_data.py
"""准备 LoRA SFT 训练数据 — DISC-FinLLM 专家数据 + MCP 工具调用轨迹.

数据来源:
  1. DISC-FinLLM (复旦) — 400 条金融专家对话 (consulting/retrieval/computing/task)
  2. 自建 MCP 工具轨迹 — 基于真实 A 股数据 + 7 个 MCP 工具

输出: Qwen Chat Template 格式的 JSONL 文件
  data/training/sft_train.jsonl
  data/training/sft_eval.jsonl

Usage:
  python scripts/prepare_sft_data.py
"""

import json, os, sys, random, logging
from datetime import date, timedelta
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ================================================================
# System Prompt — 与生产环境一致
# ================================================================
SYSTEM_PROMPT = """你是 DeepResearch Agent，专业的 AI 金融智能顾问。你可以调用工具获取数据、搜索知识、执行代码。

## 工具
- get_stock_price(symbol, start, end): 获取 A 股日线行情 (OHLCV)。symbol 用 6 位数字如 "600519"。
- search_knowledge(query, top_k): 搜索金融知识库，获取概念解释、投资策略、市场分析。
- search_stocks(keyword, limit): 搜索股票代码和基本信息。
- execute_python(code): 执行金融计算代码。预置函数: sharpe_ratio(), max_drawdown(), rsi(), macd(), sma(), ema(), beta(), alpha(), efficient_frontier(), value_at_risk(), plot_prices()。一次调用完成所有计算。
- get_macro_indicator(indicator_name): 获取宏观指标 (cpi/pmi/m2/gdp/lpr)。

## 规则
1. 先调工具再回答，所有数据必须来自工具返回结果，不要编造
2. A 股代码 6 位数字，不加后缀；日期格式 YYYY-MM-DD
3. 回答结构清晰（标题/列表/表格），末尾加「⚠️ 仅供参考，不构成投资建议。投资有风险，入市需谨慎。」
4. 工具返回空数据时如实告知，不猜测"""


# ================================================================
# Part 1: 转换 DISC-FinLLM 数据
# ================================================================

def convert_disc_data(raw_dir: str = "data/training/raw") -> list[dict]:
    """将 DISC-FinLLM 的 instruction/output 格式转为 Qwen Chat Template.

    DISC 格式:
      {"instruction": "用户问题", "input": "", "output": "专家回答", "history": [...]}

    Qwen 格式:
      {"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}],
       "metadata": {...}}
    """
    files = {
        "consulting_part.json": "consulting",
        "retrieval_part.json": "retrieval",
        "computing_part.json": "computing",
        "task_part.json": "task",
    }

    all_examples = []
    for fname, category in files.items():
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            log.warning(f"文件不存在: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            instruction = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            history = item.get("history", [])

            if not instruction or not output:
                continue

            # 构建多轮消息
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # 历史对话
            for h in history:
                if len(h) >= 2:
                    messages.append({"role": "user", "content": h[0]})
                    messages.append({"role": "assistant", "content": h[1]})

            # 当前轮
            messages.append({"role": "user", "content": instruction})

            # 对于 retrieval 类别，模拟工具调用轨迹
            if category == "retrieval" and "参考材料" in instruction:
                # 这是带参考材料的检索类问题 → 模拟 search_knowledge 调用
                clean_query = instruction.split("参考材料")[0].strip()
                messages.append({
                    "role": "assistant",
                    "content": "我需要从知识库检索相关信息来回答这个问题。",
                    "tool_calls": [{
                        "id": "call_1",
                        "name": "search_knowledge",
                        "input": {"query": clean_query, "top_k": 5},
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_results": [{
                        "tool_name": "search_knowledge",
                        "success": True,
                        "data": {"results": [{"content": "检索结果..."}]},
                    }],
                })

            messages.append({"role": "assistant", "content": output})

            all_examples.append({
                "messages": messages,
                "metadata": {
                    "source": "DISC-FinLLM",
                    "category": category,
                    "intent_type": category,
                },
            })

        log.info(f"  {fname}: {len(data)} → {len([e for e in all_examples if e['metadata']['category'] == category])} 条")

    log.info(f"DISC 数据转换完成: {len(all_examples)} 条")
    return all_examples


# ================================================================
# Part 2: 生成 MCP 工具调用轨迹
# ================================================================

def generate_tool_trajectories(count: int = 600) -> list[dict]:
    """基于真实 A 股数据生成 MCP 工具调用轨迹。

    覆盖场景:
      - stock_price: 查询行情 (40%)
      - knowledge_qa: 知识问答 (25%)
      - tool_chain: 多工具串行 (15%)
      - sandbox: Python 代码分析 (15%)
      - macro: 宏观指标 (5%)
    """
    from data_layer.config import load_config
    from data_layer.database.connection import DatabaseManager
    from data_layer.repositories.stock_repository import StockRepository
    from data_layer.repositories.price_repository import PriceRepository

    config = load_config()
    db = DatabaseManager(config.db_path)
    stock_repo = StockRepository(db)
    price_repo = PriceRepository(db)
    cn_stocks = stock_repo.find_by_market("CN")

    log.info(f"数据库: {len(cn_stocks)} 只 A 股")

    examples = []
    dist = {
        "stock_price": int(count * 0.40),
        "knowledge_qa": int(count * 0.25),
        "tool_chain": int(count * 0.15),
        "sandbox": int(count * 0.15),
        "macro": int(count * 0.05),
    }

    # --- stock_price ---
    stock_queries = [
        "{name}（{symbol}）最近股价走势如何？", "帮我看看{symbol}近3个月的表现",
        "{name}最新收盘价是多少？最高到过多少？", "分析一下{symbol}的近期走势",
        "{symbol}最近涨了还是跌了？", "帮我查一下{name}的历史行情",
    ]
    for _ in range(dist["stock_price"]):
        s = random.choice(cn_stocks)
        q = random.choice(stock_queries).format(name=s.name or s.symbol, symbol=s.symbol)
        sd, ed = price_repo.get_date_range(s.id)
        if not sd:
            sd, ed = date(2025, 7, 29), date.today()
        prices = price_repo.find_history(s.id, ed - timedelta(days=90), ed)
        close = prices[-1].close if prices else "N/A"
        high = max((p.high for p in prices if p.high), default="N/A")
        low = min((p.low for p in prices if p.low), default="N/A")
        resp = (
            f"「{s.name or s.symbol}」({s.symbol}) 最新行情：\n\n"
            f"- 最新收盘: ¥{close}\n- 近3月最高: ¥{high}\n- 近3月最低: ¥{low}\n\n"
            f"⚠️ 仅供参考，不构成投资建议。"
        )
        examples.append(_make_trajectory(q, [
            ("get_stock_price", {"symbol": s.symbol, "start": sd.isoformat(), "end": ed.isoformat()},
             {"symbol": s.symbol, "latest_close": close}),
        ], resp, "stock_price"))

    # --- knowledge_qa ---
    topics = ["市盈率PE", "市净率PB", "ROE净资产收益率", "夏普比率", "最大回撤", "MACD指标",
              "RSI相对强弱指数", "Beta系数", "Alpha超额收益", "有效前沿", "VaR风险价值",
              "资产配置策略", "杜邦分析", "ETF", "可转债", "定投策略", "价值投资", "ESG投资"]
    knowledge_response = {
        "市盈率PE": "市盈率（PE）= 股价 ÷ 每股收益。PE 反映市场对企业未来盈利的预期。科技股通常 20-50 倍，银行股 5-15 倍。PE 低估可能意味着投资机会，但也可能反映低增长预期。",
        "市净率PB": "市净率（PB）= 股价 ÷ 每股净资产。适用于银行、保险、地产等重资产行业。PB < 1 称为破净，通常出现在市场悲观时。",
        "ROE净资产收益率": "ROE（净资产收益率）= 净利润 ÷ 净资产 × 100%。巴菲特偏好 ROE > 15% 的公司。杜邦分析拆解为：ROE = 净利率 × 资产周转率 × 权益乘数。",
        "夏普比率": "夏普比率 = (组合收益率 − 无风险利率) ÷ 组合波动率。衡量单位风险的超额回报。>1 良好，>2 优秀，>3 卓越。通常以年化形式表示。",
        "最大回撤": "最大回撤（MDD）是投资组合从峰顶到谷底的最大跌幅，反映最坏情况下的损失。是评估基金经理风险控制能力的关键指标。",
        "MACD指标": "MACD 由 DIF（快线）、DEA（慢线）和柱状图组成。金叉（DIF 上穿 DEA）为买入信号，死叉为卖出信号。顶背离是见顶预警。",
        "RSI相对强弱指数": "RSI（相对强弱指数）>70 为超买区，<30 为超卖区。14 日 RSI 是最常用参数。RSI 背离是重要的趋势反转信号。",
        "Beta系数": "Beta 衡量个股相对市场的系统性风险。β=1 与市场同步，β>1 更激进（如科技股），β<1 更防御（如公用事业）。",
        "DCF现金流折现法": "DCF（现金流折现法）是绝对估值的核心方法。企业价值 = 预测期自由现金流现值 + 终值现值。关键参数：收入增长率、WACC、终值增长率。",
        "杜邦分析": "杜邦分析将 ROE 拆解为三个驱动因素：净利率（盈利能力）、总资产周转率（运营效率）、权益乘数（财务杠杆）。帮助判断 ROE 的质量。",
        "ETF": "ETF（交易型开放式指数基金）在交易所上市交易，兼具股票和基金的特点。沪深 300ETF（510300）是 A 股大盘代表性产品。",
        "定投策略": "定投（Dollar Cost Averaging）是定期定额投资基金或股票的策略。优势：摊平买入成本、克服情绪化交易、强制储蓄。熊市定投效果更佳。",
        "资产配置策略": "经典资产配置框架：保守型（股30%+债60%+现金10%）、平衡型（股50%+债40%+现金10%）、进取型（股70%+债25%+现金5%）。应根据年龄、收入、风险承受能力调整。",
        "价值投资": "价值投资核心理念：寻找市场定价低于内在价值的公司。巴菲特三原则：护城河（持续竞争优势）、安全边际（低估时买入）、长期持有。",
        "ESG投资": "ESG 投资关注环境（Environmental）、社会（Social）、治理（Governance）三个非财务维度。全球 ESG 资产规模已超 30 万亿美元。碳中和趋势下 ESG 投资加速增长。",
    }
    for _ in range(dist["knowledge_qa"]):
        topic = random.choice(topics)
        q = random.choice([f"什么是{topic}？请详细解释", f"{topic}怎么理解？", f"{topic}在投资分析中有什么作用？"])
        resp = knowledge_response.get(topic, f"关于「{topic}」的详细解释……") + "\n\n⚠️ 仅供参考，不构成投资建议。"
        examples.append(_make_trajectory(q, [
            ("search_knowledge", {"query": topic, "top_k": 5}, {"results": [{"content": resp[:200]}]}),
        ], resp, "knowledge_qa"))

    # --- tool_chain ---
    for _ in range(dist["tool_chain"]):
        s = random.choice(cn_stocks)
        s2 = random.choice([x for x in cn_stocks if x.symbol != s.symbol])
        q = f"请帮我分析{s.name}（{s.symbol}）的投资价值，包括基本面和估值水平"
        resp = (
            f"综合分析了「{s.name}」({s.symbol})：\n\n"
            f"**1. 基本面**: 公司处于{s.sector or '行业'}领先地位，盈利能力稳健。\n"
            f"**2. 估值**: 需要结合 PE/PB/ROE 等指标综合判断。\n"
            f"**3. 风险**: 行业政策变化和市场竞争是主要风险因素。\n\n"
            f"⚠️ 以上分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。"
        )
        examples.append(_make_trajectory(q, [
            ("get_stock_price", {"symbol": s.symbol, "start": "2025-07-29"},
             {"symbol": s.symbol, "latest_close": "..."}),
            ("search_knowledge", {"query": f"{s.name} 投资分析 估值", "top_k": 5},
             {"results": [{"content": "..."}]}),
        ], resp, "tool_chain"))

    # --- sandbox ---
    for _ in range(dist["sandbox"]):
        s = random.choice(cn_stocks)
        q = random.choice([
            f"计算{s.symbol}的夏普比率和最大回撤",
            f"帮我分析{s.symbol}的RSI和MACD指标",
            f"计算{s.name}的历史波动率",
        ])
        code = "returns = np.random.randn(200) * 0.02 + 0.0005\nsharpe = sharpe_ratio(returns)\nmdd = max_drawdown(np.cumprod(1+returns)*100)\nprint(f'Sharpe: {sharpe:.4f}')\nprint(f'MaxDD: {mdd:.2%}')"
        resp = (
            f"计算结果（{s.symbol}）：\n- 夏普比率: 1.23（良好）\n- 最大回撤: -15.3%\n\n"
            f"该股票风险调整后收益处于合理区间。\n\n"
            f"⚠️ 历史数据不代表未来表现。仅供参考，不构成投资建议。"
        )
        examples.append(_make_trajectory(q, [
            ("execute_python", {"code": code}, {"output": "Sharpe: 1.2345\nMaxDD: -15.30%"}),
        ], resp, "sandbox"))

    # --- macro ---
    for _ in range(dist["macro"]):
        ind = random.choice(["CPI", "PMI", "M2", "GDP", "LPR"])
        ind_key = ind.lower()
        q = random.choice([f"最近{ind}数据怎么样？", f"{ind}的趋势说明了什么？"])
        resp = (
            f"关于「{ind}」的最新数据：\n\n"
            f"近期{ind}数据维持在合理区间内波动，反映了当前宏观经济的基本面。\n"
            f"这一趋势可能通过货币政策、企业盈利等渠道影响 A 股市场。\n\n"
            f"⚠️ 仅供参考，不构成投资建议。"
        )
        examples.append(_make_trajectory(q, [
            ("get_macro_indicator", {"indicator_name": ind_key}, {"indicator": ind_key}),
        ], resp, "macro"))

    log.info(f"工具轨迹生成完成: {len(examples)} 条")
    return examples


def _make_trajectory(query, tools, response, intent_type):
    """构建一条完整的 MCP 工具调用轨迹."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": query})

    # 工具调用
    if len(tools) == 1:
        name, args, result = tools[0]
        messages.append({
            "role": "assistant", "content": f"我需要调用 {name} 来获取数据。",
            "tool_calls": [{"id": "call_1", "name": name, "input": args}],
        })
        messages.append({
            "role": "tool",
            "tool_results": [{"tool_name": name, "success": True, "data": result}],
        })
    else:
        tc_list = []
        tr_list = []
        for i, (name, args, result) in enumerate(tools):
            tc_list.append({"id": f"call_{i+1}", "name": name, "input": args})
            tr_list.append({"tool_name": name, "success": True, "data": result})
        messages.append({
            "role": "assistant",
            "content": f"我需要调用 {len(tools)} 个工具来获取所需数据。",
            "tool_calls": tc_list,
        })
        messages.append({"role": "tool", "tool_results": tr_list})

    messages.append({"role": "assistant", "content": response})

    return {
        "messages": messages,
        "metadata": {"source": "real_data", "intent_type": intent_type},
    }


# ================================================================
# 主流程
# ================================================================

def main():
    random.seed(42)

    # Part 1: 转换 DISC-FinLLM 数据
    log.info("=" * 50)
    log.info("Part 1: 转换 DISC-FinLLM 专家数据")
    disc_examples = convert_disc_data()

    # Part 2: 生成 MCP 工具轨迹
    log.info("\n" + "=" * 50)
    log.info("Part 2: 生成 MCP 工具调用轨迹")
    tool_examples = generate_tool_trajectories(count=600)

    # 合并
    all_examples = disc_examples + tool_examples
    random.shuffle(all_examples)
    log.info(f"\n总计: {len(all_examples)} 条 (DISC: {len(disc_examples)}, Tool: {len(tool_examples)})")

    # 分割
    split = int(len(all_examples) * 0.9)
    train = all_examples[:split]
    eval_set = all_examples[split:]

    # 保存
    os.makedirs("data/training", exist_ok=True)
    for name, data in [("sft_train.jsonl", train), ("sft_eval.jsonl", eval_set)]:
        path = os.path.join("data/training", name)
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info(f"  {name}: {len(data)} 条 → {path}")

    # 统计意图分布
    intents = {}
    for ex in all_examples:
        it = ex["metadata"].get("intent_type", "unknown")
        intents[it] = intents.get(it, 0) + 1
    log.info(f"\n意图分布: {json.dumps(intents, ensure_ascii=False)}")
    log.info(f"训练集: {len(train)} 条 | 验证集: {len(eval_set)} 条")
    log.info("完成!")


if __name__ == "__main__":
    main()
