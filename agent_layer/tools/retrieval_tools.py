# agent_layer/tools/retrieval_tools.py
"""多模态检索工具 — 封装 retrieval_layer 为 MCP 工具.

对应 Agent 五大工具之: 多模态智能检索工具
功能: 金融知识库语义搜索、研报搜索、政策搜索
"""

import logging
from typing import Optional

from agent_layer.mcp.registry import ToolRegistry
from agent_layer.mcp.types import ToolRole

logger = logging.getLogger(__name__)


def register_retrieval_tools(
    registry: ToolRegistry,
    search_pipeline=None,  # SearchPipeline from retrieval_layer
) -> None:
    """注册检索相关工具.

    Args:
        registry: ToolRegistry 实例.
        search_pipeline: SearchPipeline 实例 (用于语义检索).
    """

    @registry.register(
        name="search_knowledge",
        description=(
            "在金融知识库中搜索专业知识。涵盖: 股票基金基础信息、财务指标解释、"
            "投资策略、估值方法、行业分析、监管政策等。"
            "适用于: 用户询问概念解释、投资方法、行业背景等知识性问题时。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "自然语言搜索查询，如 '什么是ROE'、'市盈率估值方法'、'白酒行业分析'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认 5",
                },
                "doc_type": {
                    "type": "string",
                    "description": "文档类型过滤: stock_profile / financial_knowledge / research_report。不指定则不过滤",
                },
            },
            "required": ["query"],
        },
        role=ToolRole.RETRIEVAL,
    )
    def search_knowledge(
        query: str, top_k: int = 5, doc_type: Optional[str] = None
    ) -> dict:
        """知识库语义搜索."""
        if not search_pipeline:
            return {
                "error": "检索管道未初始化",
                "results": [],
            }

        try:
            results = search_pipeline.search(
                query,
                top_k=top_k,
                filter_doc_type=doc_type,
            )
        except Exception as e:
            logger.error("Knowledge search failed: %s", e)
            return {"error": str(e), "results": []}

        formatted = []
        for r in results:
            formatted.append({
                "score": round(r.get("score", 0), 4),
                "match_type": r.get("match_type", "unknown"),
                "content": r.get("document", "")[:500],  # 截断
                "source": {
                    "doc_type": r.get("metadata", {}).get("doc_type", ""),
                    "symbol": r.get("metadata", {}).get("symbol", ""),
                    "authority": r.get("metadata", {}).get("authority", ""),
                },
            })

        return {
            "query": query,
            "count": len(formatted),
            "results": formatted,
        }

    @registry.register(
        name="search_stock_info",
        description=(
            "搜索指定股票的详细信息，包括公司业务描述、行业分类等。"
            "适用于: 用户想了解某家公司的基本情况时。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，可以是股票代码或公司名称",
                },
            },
            "required": ["query"],
        },
        role=ToolRole.RETRIEVAL,
    )
    def search_stock_info(query: str) -> dict:
        """搜索股票详细信息 (从知识库)."""
        if not search_pipeline:
            return {"error": "检索管道未初始化", "results": []}

        results = search_pipeline.search(
            query,
            top_k=3,
            filter_doc_type="stock_profile",
        )

        formatted = []
        for r in results:
            formatted.append({
                "score": round(r.get("score", 0), 4),
                "content": r.get("document", ""),
                "symbol": r.get("metadata", {}).get("symbol", ""),
                "market": r.get("metadata", {}).get("market", ""),
                "sector": r.get("metadata", {}).get("sector", ""),
            })

        return {
            "query": query,
            "count": len(formatted),
            "results": formatted,
        }

    logger.info(
        "Registered %d retrieval tools: %s",
        2, ["search_knowledge", "search_stock_info"],
    )
