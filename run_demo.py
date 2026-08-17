#!/usr/bin/env python3
# run_demo.py
"""DeepResearch Agent 交互式 Demo — 串联全部五层.

用法:
    python run_demo.py

启动后可以输入自然语言查询，Agent 会自动：
    1. 解析意图
    2. 规划工具调用
    3. 执行工具 (数据查询 / 知识检索 / Python 沙箱)
    4. 校验结果
    5. 返回答案

示例查询:
    - 贵州茅台（600519）的股价走势如何？
    - 什么是夏普比率？如何用它来评估基金？
    - 帮我计算一个投资组合的夏普比率和最大回撤
    - 搜索新能源相关的A股
    - CPI和PMI数据说明了什么？
"""

import os
import sys
import json
import time
from datetime import date

# ── ANSI 颜色 ──────────────────────────────────────────────
C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
}


def banner():
    print(f"""
{C['cyan']}{C['bold']}
  ╔══════════════════════════════════════════════════════════╗
  ║        DeepResearch Agent — 智能金融咨询系统              ║
  ║        Data → Retrieval → Agent → Sandbox → Model        ║
  ╚══════════════════════════════════════════════════════════╝
{C['reset']}
""")


def init_system():
    """初始化所有模块。返回 orchestrator。"""
    print(f"{C['dim']}正在初始化系统...{C['reset']}")

    # ── Step 1: 数据层 ──
    print(f"  {C['green']}✓{C['reset']} 数据层: SQLite 数据库", end="")
    from data_layer.config import load_config
    from data_layer.database.connection import DatabaseManager
    from data_layer.repositories.stock_repository import StockRepository

    data_config = load_config()
    db = DatabaseManager(data_config.db_path)
    db.initialize_schema()
    stock_repo = StockRepository(db)

    # 种子数据 (如果库为空)
    if stock_repo.count() == 0:
        _seed_sample_data(stock_repo)
        print(f" (已写入示例数据)", end="")
    print()

    # ── Step 2: Agent 层 ──
    print(f"  {C['green']}✓{C['reset']} Agent 层: MCP 工具注册", end="")
    from agent_layer.config import AgentConfig
    from agent_layer.mcp.registry import ToolRegistry
    from agent_layer.llm.openai_client import OpenAIClient

    # 初始化所有 Repository
    from data_layer.repositories.price_repository import PriceRepository
    from data_layer.repositories.macro_repository import MacroRepository
    price_repo = PriceRepository(db)
    macro_repo = MacroRepository(db)

    from agent_layer.prompts.templates import build_advisor_prompt

    agent_config = AgentConfig(
        max_tool_rounds=6,
        enable_verification=True,
        enable_parallel_tools=True,
        system_prompt=build_advisor_prompt(
            cn_stocks=stock_repo.find_by_market("CN"),
            data_start="2025-07-29",
            data_end="2026-07-30",
        ),
    )
    registry = ToolRegistry()

    # ── 真实 LLM (阿里云模型服务 / OpenAI 兼容) ──
    llm_client = OpenAIClient(
        api_key="sk-ws-H.REIMXXH.EZjM.MEYCIQCHurkGqReSx0GxHsp0kSyYPoJIqrMQdu4cwaPs1pbwEwIhAIT4dgXwwrB0ktqMizptJI9oG06eJJ77agM3IT9AYHRc",
        base_url="https://ws-1tpev9v3qnsilwnf.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        temperature=0.1,
        max_tokens=4096,
    )
    print(f"\n  {C['green']}✓{C['reset']} LLM: {llm_client.model}")

    # 注册数据工具 (传入所有 repos)
    from agent_layer.tools.data_tools import register_data_tools
    register_data_tools(registry, stock_repo=stock_repo, price_repo=price_repo, macro_repo=macro_repo)
    print(f" ({registry.tool_count} tools)", end="")

    # ── 注册真实检索工具 (BGE + ChromaDB + FTS5) ──
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from retrieval_layer.embeddings.bge_embedder import BGEEmbedder
    from retrieval_layer.config import RetrievalConfig
    from retrieval_layer.stores.chroma_store import ChromaVectorStore
    from retrieval_layer.stores.fts5_store import FTS5Index
    from retrieval_layer.retrieval.dense import DenseRetriever
    from retrieval_layer.retrieval.sparse import SparseRetriever
    from retrieval_layer.pipelines.search import SearchPipeline

    ret_config = RetrievalConfig()
    embedder = BGEEmbedder('BAAI/bge-small-zh-v1.5', device='cpu')
    vector_store = ChromaVectorStore(path=ret_config.chroma_path, collection_name='finance_knowledge')
    text_store = FTS5Index(db_path=ret_config.fts5_db_path)
    search_pipeline = SearchPipeline(embedder, vector_store, text_store, ret_config)

    from agent_layer.tools.retrieval_tools import register_retrieval_tools
    register_retrieval_tools(registry, search_pipeline=search_pipeline)
    print(f" + retrieval(BGE)", end="")

    # 注册沙箱工具
    from sandbox.sandbox import register_sandbox_tool
    register_sandbox_tool(registry)
    print(f" + sandbox", end="")
    print()

    # ── Step 3: 创建 Orchestrator ──
    print(f"  {C['green']}✓{C['reset']} 调度中枢: AgentOrchestrator")
    from agent_layer.orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator(
        config=agent_config,
        registry=registry,
        llm_client=llm_client,
    )

    print(f"\n{C['bold']}就绪! 输入你的问题开始对话，输入 'quit' 退出。{C['reset']}\n")
    return orchestrator


def _seed_sample_data(stock_repo):
    """写入示例股票数据."""
    from data_layer.models.stock import Stock

    samples = [
        Stock(symbol="600519", market="CN", name="贵州茅台",
              exchange="SSE", sector="白酒", industry="食品饮料", currency="CNY"),
        Stock(symbol="000858", market="CN", name="五粮液",
              exchange="SZSE", sector="白酒", industry="食品饮料", currency="CNY"),
        Stock(symbol="300750", market="CN", name="宁德时代",
              exchange="SZSE", sector="新能源", industry="电池", currency="CNY"),
        Stock(symbol="002594", market="CN", name="比亚迪",
              exchange="SZSE", sector="新能源", industry="汽车", currency="CNY"),
        Stock(symbol="601318", market="CN", name="中国平安",
              exchange="SSE", sector="金融", industry="保险", currency="CNY"),
        Stock(symbol="600036", market="CN", name="招商银行",
              exchange="SSE", sector="金融", industry="银行", currency="CNY"),
        Stock(symbol="000001", market="CN", name="平安银行",
              exchange="SZSE", sector="金融", industry="银行", currency="CNY"),
        Stock(symbol="600276", market="CN", name="恒瑞医药",
              exchange="SSE", sector="医药", industry="制药", currency="CNY"),
        Stock(symbol="000333", market="CN", name="美的集团",
              exchange="SZSE", sector="家电", industry="制造业", currency="CNY"),
        Stock(symbol="AAPL", market="US", name="Apple Inc.",
              exchange="NASDAQ", sector="Technology", currency="USD"),
        Stock(symbol="MSFT", market="US", name="Microsoft Corp.",
              exchange="NASDAQ", sector="Technology", currency="USD"),
        Stock(symbol="NVDA", market="US", name="NVIDIA Corp.",
              exchange="NASDAQ", sector="Technology", currency="USD"),
        Stock(symbol="TSLA", market="US", name="Tesla Inc.",
              exchange="NASDAQ", sector="Automotive", currency="USD"),
        Stock(symbol="GOOGL", market="US", name="Alphabet Inc.",
              exchange="NASDAQ", sector="Technology", currency="USD"),
    ]
    for s in samples:
        stock_repo.upsert(s)


# ================================================================
# 交互式主循环
# ================================================================


def main():
    banner()

    try:
        orchestrator = init_system()
    except Exception as e:
        print(f"{C['red']}初始化失败: {e}{C['reset']}")
        import traceback
        traceback.print_exc()
        return

    # 显示可用工具
    print(f"{C['dim']}可用工具: {', '.join(orchestrator.registry.tool_names)}{C['reset']}\n")

    round_num = 0
    while True:
        try:
            query = input(f"{C['bold']}你{C['reset']}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C['dim']}再见!{C['reset']}")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q", "退出"):
            print(f"{C['dim']}再见!{C['reset']}")
            break

        round_num += 1

        # 执行
        t0 = time.time()
        try:
            response = orchestrator.run(query, use_llm=True)
        except Exception as e:
            print(f"{C['red']}执行出错: {e}{C['reset']}")
            continue

        elapsed = time.time() - t0

        # 显示结果
        intent_name = response.intent.intent_type.value if response.intent else "?"

        print(f"\n{C['cyan']}{C['bold']}Agent{C['reset']} "
              f"{C['dim']}[意图: {intent_name}, "
              f"工具: {len(response.execution.results) if response.execution else 0}个, "
              f"耗时: {elapsed:.1f}s]{C['reset']}")

        # 工具调用详情
        if response.execution:
            for r in response.execution.results:
                status = f"{C['green']}✓{C['reset']}" if r.success else f"{C['red']}✗{C['reset']}"
                print(f"  {status} {r.tool_name}", end="")
                if not r.success and r.error:
                    print(f" — {r.error[:80]}")
                else:
                    data = r.data
                    if isinstance(data, dict):
                        if "summary" in data:
                            s = data["summary"]
                            print(f" — {s.get('symbol', '')} {s.get('latest_close', '')}")
                        elif "results" in data:
                            print(f" — {data.get('count', 0)} 条结果")
                        elif "keyword" in data:
                            print(f" — 找到 {data.get('count', 0)} 只股票")
                        else:
                            print()
                    else:
                        print()

        # 答案
        print(f"\n{C['bold']}回答:{C['reset']}")
        print(f"{response.answer}\n")

        # 校验
        if response.verification:
            v = response.verification
            v_status = f"{C['green']}通过{C['reset']}" if v.passed else f"{C['yellow']}部分通过{C['reset']}"
            print(f"{C['dim']}校验: {v_status} | "
                  f"完整性:{v.dimensions[0].score:.0%} "
                  f"准确性:{v.dimensions[1].score:.0%} "
                  f"逻辑:{v.dimensions[2].score:.0%} "
                  f"合规:{v.dimensions[3].score:.0%}{C['reset']}\n")

        print(f"{C['dim']}---{C['reset']}\n")


if __name__ == "__main__":
    main()
