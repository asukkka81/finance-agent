#!/usr/bin/env python3
# app.py — DeepResearch Agent Web UI
"""Gradio 金融智能顾问 Web 界面.

Usage:
    python app.py
    → 浏览器打开 http://localhost:7860

2026-08-10: 增加思维链 (Chain-of-Thought) 展示
"""

import os, sys, time, json
from datetime import date, timedelta

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"  # 使用本地缓存，不联网检查更新

import gradio as gr

# ================================================================
# 初始化所有组件 (只执行一次)
# ================================================================

print("正在初始化系统...")

# --- 数据层 ---
from data_layer.config import load_config
from data_layer.database.connection import DatabaseManager
from data_layer.repositories.stock_repository import StockRepository
from data_layer.repositories.price_repository import PriceRepository
from data_layer.repositories.macro_repository import MacroRepository

data_config = load_config()
db = DatabaseManager(data_config.db_path)
db.initialize_schema()
stock_repo = StockRepository(db)
price_repo = PriceRepository(db)
macro_repo = MacroRepository(db)

cn_stocks = stock_repo.find_by_market("CN")
stock_list_str = ", ".join(f"{s.symbol} {s.name}" for s in cn_stocks)
min_date = "2025-07-29"
max_date = date.today().isoformat()

print(f"  数据层: {stock_repo.count()} 只股票, {price_repo.count()} 条行情")

# --- Agent 层 ---
from agent_layer.config import AgentConfig
from agent_layer.mcp.registry import ToolRegistry
from agent_layer.llm.openai_client import OpenAIClient
from agent_layer.orchestrator import AgentOrchestrator
from agent_layer.prompts.templates import build_advisor_prompt

registry = ToolRegistry()

# 数据工具
from agent_layer.tools.data_tools import register_data_tools
register_data_tools(registry, stock_repo=stock_repo, price_repo=price_repo, macro_repo=macro_repo)

# 检索工具
print(f"  Agent 层: 加载检索 ...", flush=True)
from retrieval_layer.embeddings.bge_embedder import BGEEmbedder
from retrieval_layer.config import RetrievalConfig
from retrieval_layer.stores.chroma_store import ChromaVectorStore
from retrieval_layer.stores.fts5_store import FTS5Index
from retrieval_layer.pipelines.search import SearchPipeline
from agent_layer.tools.retrieval_tools import register_retrieval_tools

ret_config = RetrievalConfig()
embedder = BGEEmbedder("BAAI/bge-small-zh-v1.5", device="cpu")
print(f"  Agent 层: BGE 就绪, 加载 ChromaDB ...", flush=True)
vector_store = ChromaVectorStore(path=ret_config.chroma_path, collection_name="finance_knowledge")
print(f"  Agent 层: ChromaDB 就绪 ({vector_store.count()} docs), 加载 FTS5 ...", flush=True)
text_store = FTS5Index(db_path=ret_config.fts5_db_path)
print(f"  Agent 层: FTS5 就绪 ({text_store.count()} docs), 构建 SearchPipeline ...", flush=True)
search_pipeline = SearchPipeline(embedder, vector_store, text_store, ret_config)
print(f"  Agent 层: SearchPipeline 就绪", flush=True)
register_retrieval_tools(registry, search_pipeline=search_pipeline)

# 实时数据工具
print(f"  Agent 层: 注册实时数据工具 ...", flush=True)
from agent_layer.tools.realtime_tools import register_realtime_tools
register_realtime_tools(registry)

# 沙箱工具
from sandbox.sandbox import register_sandbox_tool
register_sandbox_tool(registry)

print(f"  Agent 层: {registry.tool_count} 个 MCP 工具")

# LLM
print(f"  Agent 层: 初始化 LLM ...", flush=True)
llm_client = OpenAIClient(
    api_key="sk-ws-H.REIMXXH.EZjM.MEYCIQCHurkGqReSx0GxHsp0kSyYPoJIqrMQdu4cwaPs1pbwEwIhAIT4dgXwwrB0ktqMizptJI9oG06eJJ77agM3IT9AYHRc",
    base_url="https://ws-1tpev9v3qnsilwnf.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus", temperature=0.1, max_tokens=4096,
)

# Orchestrator
agent_config = AgentConfig(
    max_tool_rounds=6, enable_verification=True,
    system_prompt=build_advisor_prompt(cn_stocks=cn_stocks, data_start=min_date, data_end=max_date),
)
orchestrator = AgentOrchestrator(config=agent_config, registry=registry, llm_client=llm_client)

print(f"  就绪! {len(cn_stocks)} 只 A 股, {price_repo.count()} 条行情, {registry.tool_count} 个工具\n")


# ================================================================
# 思维链 HTML 渲染
# ================================================================

def render_chain_of_thought(chain_of_thought: list[dict], elapsed: float) -> str:
    """将思维链渲染为可折叠的 HTML 面板.

    Args:
        chain_of_thought: orchestrator 返回的思维链记录.
        elapsed: 总耗时 (秒).

    Returns:
        HTML 字符串.
    """
    if not chain_of_thought:
        return ""

    rounds_html = []
    tool_count = 0

    for step in chain_of_thought:
        round_num = step.get("round", 0)
        step_type = step.get("type", "")
        thinking = step.get("thinking", "").strip()
        tools = step.get("tools", [])

        # 每轮的徽章样式
        if step_type == "tool_call":
            badge_color = "#3b82f6"  # blue
            badge_text = f"🔧 第 {round_num} 轮"
        elif step_type == "final_answer":
            badge_color = "#10b981"  # green
            badge_text = f"📝 生成回答"
        elif step_type == "timeout":
            badge_color = "#f59e0b"  # amber
            badge_text = f"⏱ 强制结束"
        else:
            badge_color = "#6b7280"
            badge_text = f"第 {round_num} 轮"

        step_html = f'<div style="margin-bottom:8px;border-left:3px solid {badge_color};padding-left:8px;">'

        # 轮次标题
        step_html += (
            f'<span style="display:inline-block;background:{badge_color};color:#fff;'
            f'padding:1px 8px;border-radius:10px;font-size:0.85em;margin-bottom:4px;">'
            f'{badge_text}</span> '
        )

        # LLM 思考内容
        if thinking:
            # 截断过长的思考
            display_thinking = thinking[:300] + "…" if len(thinking) > 300 else thinking
            step_html += (
                f'<div style="font-size:0.85em;color:#6b7280;font-style:italic;'
                f'margin:2px 0;">💭 {display_thinking}</div>'
            )

        # 工具调用列表
        if tools:
            step_html += '<div style="font-size:0.85em;margin:4px 0;">'
            for t in tools:
                tool_count += 1
                icon = "✅" if t.get("success") else "❌"
                args_str = json.dumps(t.get("arguments", {}), ensure_ascii=False)
                if len(args_str) > 60:
                    args_str = args_str[:60] + "…"
                step_html += (
                    f'<div style="padding:1px 0;">'
                    f'{icon} <code style="font-size:0.9em;">{t["name"]}</code>'
                    f'<span style="color:#9ca3af;font-size:0.8em;">({args_str})</span>'
                    f'</div>'
                )
            step_html += '</div>'

        step_html += '</div>'
        rounds_html.append(step_html)

    # 汇总面板
    summary_line = f"🧠 思考过程 &nbsp;·&nbsp; {len(chain_of_thought)} 轮 &nbsp;·&nbsp; {tool_count} 个工具 &nbsp;·&nbsp; {elapsed:.1f}s"

    html = (
        f'<details open style="margin:8px 0;font-size:0.9em;">'
        f'<summary style="cursor:pointer;font-weight:600;color:#374151;">{summary_line}</summary>'
        f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;'
        f'padding:10px 14px;margin-top:6px;max-height:360px;overflow-y:auto;">'
        f'{"".join(rounds_html)}'
        f'</div>'
        f'</details>'
    )
    return html


# ================================================================
# 流式对话处理 — 边思考边渲染
# ================================================================

def chat_fn(message, history):
    """处理每条用户消息，流式展示思维链."""
    if not message.strip():
        return "", history

    t0 = time.time()

    # 初始化历史记录
    new_history = list(history) if history else []
    new_history.append({
        "role": "user",
        "content": [{"text": message, "type": "text"}],
    })

    # 助手回答缓冲区
    assistant_parts = []
    current_cot_html = ""

    # 流式接收 orchestrator 的每一步
    for event in orchestrator.run_stream(message, use_llm=True):
        etype = event.get("type")

        if etype == "thinking":
            # 更新思考状态
            content = event.get("content", "")
            round_num = event.get("round", 0)
            # 累积到一个可见的 CoT 面板中
            cot_lines = []
            # 重建当前 CoT
            cot_lines.append(
                '<details open style="margin:4px 0;font-size:0.9em;">'
                '<summary style="cursor:pointer;font-weight:600;color:#374151;">'
                f'🧠 正在思考… (第 {round_num} 轮)'
                '</summary>'
                '<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;'
                'padding:8px 12px;margin-top:4px;max-height:240px;overflow-y:auto;">'
            )
            cot_lines.append(
                f'<div style="font-size:0.85em;color:#6b7280;font-style:italic;">'
                f'💭 {content}</div>'
            )
            cot_lines.append('</div></details>')
            current_cot_html = "".join(cot_lines)

        elif etype == "tool_call":
            # 工具调用中
            tool_name = event.get("tool_name", "")
            args = event.get("arguments", {})
            args_str = json.dumps(args, ensure_ascii=False)
            if len(args_str) > 50:
                args_str = args_str[:50] + "…"

            cot_lines = []
            cot_lines.append(
                '<details open style="margin:4px 0;font-size:0.9em;">'
                '<summary style="cursor:pointer;font-weight:600;color:#374151;">'
                f'🧠 思考过程'
                '</summary>'
                '<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;'
                'padding:8px 12px;margin-top:4px;max-height:280px;overflow-y:auto;">'
            )
            cot_lines.append(
                f'<div style="padding:2px 0;font-size:0.85em;">'
                f'🔄 调用 <code style="font-size:0.9em;">{tool_name}</code>'
                f'<span style="color:#9ca3af;font-size:0.8em;">({args_str})</span>'
                f'</div>'
            )
            cot_lines.append('</div></details>')
            current_cot_html = "".join(cot_lines)

        elif etype == "tool_result":
            # 工具结果
            tool_name = event.get("tool_name", "")
            success = event.get("success", False)
            summary = event.get("summary", "")
            icon = "✅" if success else "❌"
            if len(summary) > 80:
                summary = summary[:80] + "…"

            cot_lines = []
            cot_lines.append(
                '<details open style="margin:4px 0;font-size:0.9em;">'
                '<summary style="cursor:pointer;font-weight:600;color:#374151;">'
                f'🧠 思考过程'
                '</summary>'
                '<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;'
                'padding:8px 12px;margin-top:4px;max-height:280px;overflow-y:auto;">'
            )
            cot_lines.append(
                f'<div style="padding:2px 0;font-size:0.85em;">'
                f'{icon} <code style="font-size:0.9em;">{tool_name}</code>'
                f' → {summary}'
                f'</div>'
            )
            cot_lines.append('</div></details>')
            current_cot_html = "".join(cot_lines)

        elif etype == "answer":
            # 最终回答
            elapsed = event.get("elapsed", time.time() - t0)
            answer = event.get("content", "")
            break

        # 流式输出中间状态: CoT + 占位文字
        placeholder = "" if etype == "answer" else "⏳ 分析中…"
        display_msg = current_cot_html + "\n\n" + placeholder if current_cot_html else placeholder

        yield "", new_history + [{
            "role": "assistant",
            "content": [{"text": display_msg, "type": "text"}],
        }]
    else:
        # 如果没有 answer event，用空回答
        answer = "抱歉，未能生成回答。"
        elapsed = time.time() - t0

    # 最终输出
    elapsed = time.time() - t0
    final_cot_html = (
        f'<details style="margin:4px 0;font-size:0.9em;">'
        f'<summary style="cursor:pointer;font-weight:600;color:#10b981;">'
        f'🧠 思考完成 · {elapsed:.1f}s'
        f'</summary>'
        f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;'
        f'padding:8px 12px;margin-top:4px;max-height:240px;overflow-y:auto;font-size:0.85em;">'
        f'{current_cot_html}'
        f'</div>'
        f'</details>'
    )

    final_msg = final_cot_html + "\n\n" + answer

    yield "", new_history + [{
        "role": "assistant",
        "content": [{"text": final_msg, "type": "text"}],
    }]

# 原来的同步版本留作备用
def chat_fn_sync(message, history):
    """非流式对话 (备用)."""
    if not message.strip():
        return "", history

    t0 = time.time()
    response = orchestrator.run(message, use_llm=True)
    elapsed = time.time() - t0

    parts = []

    if hasattr(response, 'chain_of_thought') and response.chain_of_thought:
        cot_html = render_chain_of_thought(response.chain_of_thought, elapsed)
        if cot_html:
            parts.append(cot_html)

    if response.answer:
        parts.append(response.answer)
    else:
        parts.append("*未能生成回答，请重试。*")

    assistant_msg = "\n\n".join(parts)

    history = []
    history.append({
        "role": "user",
        "content": [{"text": message, "type": "text"}],
    })
    history.append({
        "role": "assistant",
        "content": [{"text": assistant_msg, "type": "text"}],
    })
    return "", history


# ================================================================
# Gradio UI
# ================================================================

def build_ui():
    css = """
    .status-box { padding: 12px; border-radius: 8px; margin: 4px 0; }
    .tool-row { font-family: monospace; font-size: 0.9em; padding: 2px 0; }
    footer { display: none !important; }
    details summary { cursor: pointer; user-select: none; }
    details summary:hover { color: #1d4ed8; }
    """

    with gr.Blocks(title="DeepResearch Agent — 智能金融顾问") as demo:

        # --- Header ---
        gr.Markdown(
            "# 🏦 DeepResearch Agent\n"
            "### AI 金融智能顾问 — 实时数据 · 知识检索 · 量化分析"
        )

        with gr.Row():
            # --- Sidebar (状态) ---
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### 📊 系统状态")

                status_html = (
                    f"<div style='font-size:0.9em; line-height:1.8;'>"
                    f"🟢 <b>LLM:</b> 通义千问 (qwen-plus)<br>"
                    f"🟢 <b>数据:</b> {price_repo.count()} 条 A 股行情<br>"
                    f"🟢 <b>股票:</b> {len(cn_stocks)} 只<br>"
                    f"🟢 <b>工具:</b> {registry.tool_count} 个 MCP<br>"
                    f"🟢 <b>检索:</b> BGE + ChromaDB (49 篇)<br>"
                    f"🟡 <b>Reranker:</b> 未加载 (磁盘不足)<br>"
                    f"🔴 <b>美股:</b> 无数据<br>"
                    f"</div>"
                )
                gr.HTML(status_html)

                gr.Markdown("### 📋 可用股票")
                stock_md = "\n".join(
                    f"- **{s.symbol}** {s.name} ({s.sector or '综合'})"
                    for s in cn_stocks
                )
                gr.Markdown(stock_md + f"\n\n*{min_date} ~ {max_date}*")

                gr.Markdown("### 💡 试试这些")
                examples = gr.Examples(
                    examples=[
                        "贵州茅台（600519）最新股价和走势如何？",
                        "什么是夏普比率？和索提诺比率有什么区别？",
                        "五粮液和茅台哪只股票波动更大？请用数据说明",
                        "用 Python 计算宁德时代的 RSI 和 MACD 指标",
                        "帮我分析招商银行的投资价值",
                        "资产配置应该怎么做？保守型投资者怎么配？",
                    ],
                    inputs=[gr.State()],
                    label="",
                )

            # --- Main Chat ---
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    height=600,
                    placeholder="输入你的金融问题...",
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="例如：贵州茅台最近涨了还是跌了？",
                        scale=9,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                gr.Markdown(
                    "<div style='text-align:center;color:#888;font-size:0.8em;'>"
                    "⚠️ 所有分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。"
                    "</div>"
                )

        # --- Events ---
        msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
        send_btn.click(chat_fn, [msg, chatbot], [msg, chatbot])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    )
