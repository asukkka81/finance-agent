# agent_layer/tools/web_tools.py
"""网页内容抓取工具 — Web 搜索 & 内容抓取.

对应 Agent 五大工具之: 网页内容抓取工具
功能: 网页搜索、URL 内容抓取、权威网站定向获取。

用途: 补充实时信息 (财报发布、政策更新、市场新闻)，
      这些信息可能不在本地知识库中。
"""

import logging
import re
from typing import Optional

from agent_layer.mcp.registry import ToolRegistry
from agent_layer.mcp.types import ToolRole

logger = logging.getLogger(__name__)


def register_web_tools(registry: ToolRegistry) -> None:
    """注册 Web 相关工具."""

    @registry.register(
        name="web_search",
        description=(
            "搜索互联网获取最新的金融资讯、公告、新闻。"
            "当用户询问的信息具有时效性 (如最新财报、政策变化、今日行情分析) 时使用。"
            "返回搜索结果摘要，可进一步用 web_fetch 获取详细内容。"
            "注意: 此工具有调用成本，仅在需要最新信息时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，如 '贵州茅台 2024年财报'、'央行降息最新消息'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大结果数，默认 5",
                },
            },
            "required": ["query"],
        },
        role=ToolRole.WEB_FETCH,
    )
    def web_search(query: str, max_results: int = 5) -> dict:
        """网页搜索.

        实际实现需要对接搜索 API (如 Bing/SERP API)。
        当前为框架实现，返回占位符。
        """
        logger.info("Web search: %s (max=%d)", query, max_results)

        # 框架实现 — 实际对接搜索 API 时替换
        return {
            "query": query,
            "source": "web_search_placeholder",
            "count": 0,
            "results": [],
            "message": (
                "网页搜索功能已注册，需要配置搜索 API (如 Bing Search API) "
                "才能获取真实结果。请设置环境变量 SEARCH_API_KEY。"
            ),
        }

    @registry.register(
        name="web_fetch",
        description=(
            "抓取指定 URL 的网页内容，提取正文用于分析。"
            "适用于: 获取证监会公告、交易所公示、金融机构研究报告等权威网页内容。"
            "限制: 仅抓取公开可访问的网页，不处理需要登录的页面。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "目标网页 URL",
                },
                "max_length": {
                    "type": "integer",
                    "description": "提取文本的最大长度 (字符)，默认 5000",
                },
            },
            "required": ["url"],
        },
        role=ToolRole.WEB_FETCH,
    )
    def web_fetch(url: str, max_length: int = 5000) -> dict:
        """抓取网页内容."""
        logger.info("Web fetch: %s", url)

        # 基本 URL 校验
        if not re.match(r'^https?://', url):
            return {"error": f"无效的 URL: {url}", "content": ""}

        try:
            import requests
            resp = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "FinanceAgent/1.0"
                    ),
                },
            )
            resp.raise_for_status()

            # 简单提取正文 (去除 HTML 标签)
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) > max_length:
                text = text[:max_length] + f"\n... (截断，原文共 {len(text)} 字符)"

            return {
                "url": url,
                "status_code": resp.status_code,
                "content": text,
                "length": len(text),
            }

        except ImportError:
            return {
                "error": "requests 库未安装",
                "content": "",
                "hint": "pip install requests",
            }
        except Exception as e:
            logger.warning("Web fetch failed for %s: %s", url, e)
            return {
                "url": url,
                "error": str(e),
                "content": "",
            }

    logger.info(
        "Registered %d web tools: %s",
        2, ["web_search", "web_fetch"],
    )
