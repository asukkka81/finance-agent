# retrieval_layer/indexer/splitter.py
"""文本分块器 — 将长文档切分为适合向量检索的 chunk."""

import logging
import re
from typing import Optional

from retrieval_layer.indexer.loader import Document

logger = logging.getLogger(__name__)


class TextSplitter:
    """中文友好的文本分块器.

    策略:
        1. 按段落 (\\n\\n) 分割
        2. 保证语义完整性: 段落不被截断
        3. chunk_overlap 确保相邻块有上下文重叠
        4. 最小 chunk 长度: 过滤过短的片段

    Usage::

        splitter = TextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split_documents(docs)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 20,
        separators: Optional[list[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or [
            "\n\n",     # 段落
            "\n",       # 换行
            "。",       # 中文句号
            "；",       # 中文分号
            ". ",       # 英文句号
            "! ",       # 英文感叹号
            "? ",       # 英文问号
            " ",        # 空格 (最后手段)
        ]

    # ================================================================
    # Document 级分块
    # ================================================================

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """将 Document 列表按 chunk_size 分块.

        每块继承原文档的 metadata，并添加 chunk 索引信息。
        """
        chunks = []
        for doc in documents:
            doc_chunks = self.split_text(doc.content)
            for i, chunk_text in enumerate(doc_chunks):
                chunk_meta = doc.metadata.copy()
                chunk_meta["chunk_index"] = i
                chunk_meta["chunk_total"] = len(doc_chunks)
                if doc.metadata.get("source_file"):
                    chunk_meta["source_file"] = doc.metadata["source_file"]

                chunk = Document(
                    content=chunk_text,
                    metadata=chunk_meta,
                )
                chunks.append(chunk)

        logger.debug(
            "Split %d documents into %d chunks", len(documents), len(chunks)
        )
        return chunks

    # ================================================================
    # 文本级分块
    # ================================================================

    def split_text(self, text: str) -> list[str]:
        """将长文本分块 (递归分隔符策略).

        核心算法:
            1. 如果 text 长度 ≤ chunk_size → 直接返回
            2. 尝试用当前分隔符分割
            3. 合并片段，保证每块接近 chunk_size 且有 overlap
            4. 如果分隔符分割后仍有超长片段，换下一个分隔符递归
        """
        if len(text) <= self.chunk_size:
            # 短文本直接返回 (不受 min_chunk_size 限制)
            return [text] if text.strip() else []

        return self._split_recursive(text, self.separators.copy())

    def _split_recursive(
        self, text: str, separators: list[str]
    ) -> list[str]:
        """递归分块."""
        if not separators:
            # 没有分隔符了，直接按长度截断
            return self._split_by_length(text)

        sep = separators[0]
        remaining = separators[1:]

        # 用当前分隔符切分
        parts = re.split(f"({re.escape(sep)})", text)

        # 合并分隔符到前面的片段
        merged = []
        for part in parts:
            if merged and part == sep:
                merged[-1] += sep
            else:
                merged.append(part)

        # 构建 chunks
        chunks = []
        current_chunk = ""

        for part in merged:
            # 如果单个片段就超过了 chunk_size，递归处理
            if len(part) > self.chunk_size:
                # 先保存 current_chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # 递归
                sub_chunks = self._split_recursive(part, remaining)
                chunks.extend(sub_chunks)
                current_chunk = ""
                continue

            # 加入当前 chunk 后会超长？
            if len(current_chunk) + len(part) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # 保留 overlap 部分
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = overlap_text + part
            else:
                current_chunk += part

        # 最后一个 chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 过滤太短的
        return [c for c in chunks if len(c.strip()) >= self.min_chunk_size]

    def _split_by_length(self, text: str) -> list[str]:
        """纯长度切分 (最后手段)."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
        return chunks
