# retrieval_layer/utils/text_utils.py
"""文本处理工具."""

import re


def normalize_query(query: str) -> str:
    """查询文本规范化.

    操作:
        1. 去除多余空白
        2. 统一标点符号
    """
    if not query:
        return ""

    # 去多余空白
    query = re.sub(r"\s+", " ", query).strip()

    # 统一全角标点
    punctuation_map = {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "；": ";",
        "：": ":",
        """: "\"",
        """: "\"",
        "（": "(",
        "）": ")",
    }
    for full, half in punctuation_map.items():
        query = query.replace(full, half)

    return query


def truncate_text(text: str, max_length: int = 200) -> str:
    """截断文本用于展示."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """简单关键词提取 (基于 TF).

    用于查询日志 & 调试，不做为核心功能。
    """
    # 去除标点 & 空白
    words = re.findall(r"[一-鿿]+|[a-zA-Z]+", text)

    # 统计词频
    freq: dict[str, int] = {}
    for w in words:
        if len(w) >= 2:  # 过滤单字
            freq[w] = freq.get(w, 0) + 1

    # 按频次排序
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def format_search_result(result: dict, max_length: int = 300) -> str:
    """格式化单条搜索结果为可读字符串.

    供 Agent 在对话中将检索结果展示给用户。
    """
    lines = [
        f"[{result.get('match_type', 'unknown')}] "
        f"Score: {result.get('score', 0):.4f}",
    ]

    doc = result.get("document", "")
    if len(doc) > max_length:
        doc = doc[:max_length] + "..."

    lines.append(f"Content: {doc}")

    meta = result.get("metadata", {})
    if meta:
        # 只显示关键元数据
        key_meta = {
            k: v for k, v in meta.items()
            if k in ("doc_type", "symbol", "market", "sector", "authority", "source_file")
        }
        if key_meta:
            lines.append(f"Meta: {key_meta}")

    return "\n".join(lines)
