#!/usr/bin/env python3
"""Agent 工具调度评测 — 10 问全轨迹记录.

工具集: get_stock_price / search_knowledge / search_stocks / get_macro_indicator /
        execute_python (沙箱) / execute_sql (Text2SQL)
模型: qwen-plus (阿里云 MAAS, 与 app.py 相同配置)
轨迹: 每问完整记录 意图 → 每轮 LLM 决策/工具调用/工具结果 → 最终校验 → 答案

输出: data/test_logs/agent_eval_<日期>.jsonl (结构化) + .txt (人读版)
用法: python scripts/run_agent_eval.py [--limit N] [--questions q1,q2,...]
"""

import argparse
import json
import socket
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
socket.setdefaulttimeout(120)  # 防止单次网络调用无限挂起
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_layer.config import load_config
from data_layer.database.connection import DatabaseManager
from data_layer.repositories.stock_repository import StockRepository
from data_layer.repositories.price_repository import PriceRepository
from data_layer.repositories.macro_repository import MacroRepository

from agent_layer.config import AgentConfig
from agent_layer.llm.openai_client import OpenAIClient
from agent_layer.mcp.registry import ToolRegistry
from agent_layer.orchestrator import AgentOrchestrator
from agent_layer.prompts.templates import build_advisor_prompt
from agent_layer.tools.data_tools import register_data_tools, register_sql_tool
from agent_layer.tools.retrieval_tools import register_retrieval_tools

from retrieval_layer.config import RetrievalConfig
from retrieval_layer.embeddings.bge_embedder import BGEEmbedder
from retrieval_layer.pipelines.search import SearchPipeline
from retrieval_layer.stores.chroma_store import ChromaVectorStore
from retrieval_layer.stores.fts5_store import FTS5Index

from sandbox.sandbox import register_sandbox_tool

QUESTIONS = [
    # --- 行情直查 ---
    "贵州茅台最近30个交易日的涨跌幅是多少？",
    # --- 沙箱计算 (需先取行情 → 工具链) ---
    "请计算贵州茅台过去一年的年化波动率。",
    # --- RAG 知识 ---
    "什么是夏普比率？应该如何解读这个指标？",
    # --- Text2SQL ---
    "2024年涨幅最高的股票是哪只？涨幅是多少？",
    # --- 跨股票对比 ---
    "对比贵州茅台和五粮液2025年的涨跌幅表现。",
    # --- Text2SQL 统计 ---
    "中国平安2024年成交量最大的交易日是哪一天？成交量是多少？",
    # --- 沙箱计算 (最大回撤) ---
    "宁德时代过去两年的最大回撤是多少？",
    # --- 综合分析 (多工具) ---
    "请分析白酒行业当前的投资价值。",
    # --- 混合: 计算 + 解读 ---
    "计算招商银行2025年日涨跌幅的标准差，并解释这个数字的含义。",
    # --- 技术指标 ---
    "用20日均线分析贵州茅台近期的价格趋势。",
]


def build_agent():
    """镜像 app.py 的工具与 LLM 接线 (不含实时行情工具, 东财接口受限)."""
    config = load_config()
    db = DatabaseManager(config.db_path)
    stock_repo = StockRepository(db)
    price_repo = PriceRepository(db)
    macro_repo = MacroRepository(db)
    cn_stocks = stock_repo.find_by_market("CN")

    # 数据范围 (来自实际库)
    with db.get_connection() as conn:
        data_min, data_max = conn.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM daily_prices"
        ).fetchone()
    data_min, data_max = str(data_min)[:10], str(data_max)[:10]
    print(f"数据层: {len(cn_stocks)} 只股票, {price_repo.count()} 条行情 ({data_min} ~ {data_max})")

    registry = ToolRegistry()
    register_data_tools(registry, stock_repo=stock_repo, price_repo=price_repo, macro_repo=macro_repo)
    register_sql_tool(registry, db_path=config.db_path)

    ret_config = RetrievalConfig()
    embedder = BGEEmbedder("BAAI/bge-small-zh-v1.5", device="cpu")
    vector_store = ChromaVectorStore(path=ret_config.chroma_path, collection_name="finance_knowledge")
    text_store = FTS5Index(db_path=ret_config.fts5_db_path)
    search_pipeline = SearchPipeline(embedder, vector_store, text_store, ret_config)
    register_retrieval_tools(registry, search_pipeline=search_pipeline)

    register_sandbox_tool(registry)
    print(f"工具注册完成: {registry.tool_count} 个: {sorted(registry.tool_names)}")

    llm_client = OpenAIClient(
        api_key="sk-ws-H.REIMXXH.EZjM.MEYCIQCHurkGqReSx0GxHsp0kSyYPoJIqrMQdu4cwaPs1pbwEwIhAIT4dgXwwrB0ktqMizptJI9oG06eJJ77agM3IT9AYHRc",
        base_url="https://ws-1tpev9v3qnsilwnf.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus", temperature=0.1, max_tokens=4096,
    )

    agent_config = AgentConfig(
        max_tool_rounds=6, enable_verification=True,
        system_prompt=build_advisor_prompt(cn_stocks=cn_stocks, data_start=data_min, data_end=data_max),
    )
    return AgentOrchestrator(config=agent_config, registry=registry, llm_client=llm_client)


def serialize_step(step: dict) -> dict:
    """chain_of_thought 步骤序列化 (含完整工具参数与结果摘要)."""
    out = dict(step)
    tools = []
    for t in step.get("tools", []):
        tools.append({
            "name": t.get("name"),
            "arguments": t.get("arguments"),
            "success": t.get("success"),
            "summary": str(t.get("summary", ""))[:500],
        })
    out["tools"] = tools
    return out


def run_one(agent: AgentOrchestrator, question: str, qid: int) -> dict:
    """执行单问并收集全轨迹."""
    print(f"\n{'='*70}\n[{qid}] {question}")
    t0 = datetime.now()
    try:
        resp = agent.run(question, use_llm=True)
    except Exception as e:
        return {
            "id": qid, "question": question, "status": "ERROR",
            "error": f"{type(e).__name__}: {e}", "elapsed_seconds": 0,
        }

    trace = {
        "id": qid,
        "question": question,
        "status": "OK",
        "intent": {
            "type": resp.intent.intent_type.value,
            "entities": resp.intent.entities,
            "confidence": resp.intent.confidence,
        },
        "rounds": [serialize_step(s) for s in (resp.chain_of_thought or [])],
        "tool_results_full": [
            {
                "tool_name": r.tool_name,
                "success": r.success,
                "data": json.loads(json.dumps(r.data, default=str))
                if not isinstance(r.data, (str, int, float, type(None))) else r.data,
                "error": r.error,
            }
            for r in (resp.execution.results if resp.execution else [])
        ],
        "verification": (
            {
                "passed": resp.verification.passed,
                "overall_score": round(resp.verification.overall_score, 4),
                "dimensions": [
                    {
                        "dimension": d.dimension,
                        "verdict": d.verdict.value,
                        "score": round(d.score, 4),
                        "issues": d.issues,
                        "suggestions": d.suggestions,
                    }
                    for d in resp.verification.dimensions
                ],
            }
            if resp.verification else None
        ),
        "answer": resp.answer,
        "elapsed_seconds": round(resp.elapsed_seconds, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"  耗时 {trace['elapsed_seconds']}s | 校验 {trace['verification']['overall_score'] if trace['verification'] else 'N/A'} | 答案 {len(resp.answer)} 字")
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 问 (0=全部)")
    parser.add_argument("--questions", type=str, default="", help="逗号分隔的问题编号, 如 1,3,5")
    args = parser.parse_args()

    agent = build_agent()

    if args.questions:
        idxs = {int(x) - 1 for x in args.questions.split(",") if x.strip().isdigit()}
    else:
        idxs = set(range(len(QUESTIONS)))
    if args.limit:
        idxs = {i for i in sorted(idxs)[:args.limit]}

    out_dir = Path("data/test_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"agent_eval_{stamp}.jsonl"
    txt_path = out_dir / f"agent_eval_{stamp}.txt"

    print(f"\n开始评测: {len(idxs)} 个问题 → {jsonl_path}")

    traces = []
    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for qid in sorted(idxs):
            trace = run_one(agent, QUESTIONS[qid], qid + 1)
            traces.append(trace)
            jf.write(json.dumps(trace, ensure_ascii=False) + "\n")
            jf.flush()

    # 人读版
    with open(txt_path, "w", encoding="utf-8") as tf:
        for t in traces:
            tf.write(f"{'='*80}\n")
            tf.write(f"[{t['id']}] {t['question']}  (status={t['status']}, {t['elapsed_seconds']}s)\n")
            if t["status"] != "OK":
                tf.write(f"ERROR: {t.get('error')}\n\n")
                continue
            tf.write(f"意图: {t['intent']['type']} | 实体: {json.dumps(t['intent']['entities'], ensure_ascii=False)}\n")
            for s in t["rounds"]:
                if s["type"] == "tool_call":
                    tf.write(f"\n  [轮 {s['round']}] 思考: {s.get('thinking','')[:200]}\n")
                    for tool in s["tools"]:
                        tf.write(f"    → {tool['name']}({json.dumps(tool['arguments'], ensure_ascii=False, default=str)[:200]}) "
                                 f"success={tool['success']}\n")
                        tf.write(f"      结果: {tool['summary'][:300]}\n")
                elif s["type"] in ("final_answer", "timeout"):
                    tf.write(f"\n  [{s['type']} 轮 {s['round']}] {s.get('thinking','')[:200]}\n")
                else:
                    tf.write(f"\n  [轮 {s['round']}] {s.get('thinking','')[:200]}\n")
            if t["verification"]:
                v = t["verification"]
                tf.write(f"\n  校验: passed={v['passed']} overall={v['overall_score']}\n")
                for d in v["dimensions"]:
                    tf.write(f"    - {d['dimension']}: {d['verdict']} {d['score']} {d['issues']}\n")
            tf.write(f"\n  回答:\n{t['answer']}\n\n")

    print(f"\n评测完成: {jsonl_path} / {txt_path}")


if __name__ == "__main__":
    main()
