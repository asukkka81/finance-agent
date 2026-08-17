# retrieval_layer/indexer/loader.py
"""文档加载器 — 从多种来源加载文档."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """统一文档对象.

    Attributes:
        id: 唯一标识 (默认用内容的 SHA256 前 16 位).
        content: 文本内容.
        metadata: 元数据 (来源/类型/时效/权威度).
    """

    content: str
    metadata: dict = field(default_factory=dict)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()[:16]
        if "indexed_at" not in self.metadata:
            self.metadata["indexed_at"] = datetime.now().isoformat()


class DocumentLoader:
    """文档加载器 — 从多种来源加载文档.

    支持的格式:
        - .txt / .md 纯文本
        - .json / .jsonl 结构化数据
        - 从 data_layer 导入股票/基金信息
        - 从目录递归加载
    """

    def __init__(self):
        self._handlers = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".json": self._load_json,
            ".jsonl": self._load_jsonl,
        }

    # ================================================================
    # 文件加载
    # ================================================================

    def load_file(
        self, path: str, metadata: Optional[dict] = None
    ) -> list[Document]:
        """从单个文件加载文档.

        Args:
            path: 文件路径.
            metadata: 额外元数据 (会合并到每个文档).

        Returns:
            Document 列表.
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("File not found: %s", path)
            return []

        suffix = file_path.suffix.lower()
        handler = self._handlers.get(suffix, self._load_text)

        docs = handler(file_path)
        if metadata:
            for doc in docs:
                doc.metadata.update(metadata)
                doc.metadata["source_file"] = str(file_path)

        logger.debug("Loaded %d docs from %s", len(docs), path)
        return docs

    def load_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        metadata: Optional[dict] = None,
    ) -> list[Document]:
        """从目录递归加载所有支持的文档.

        Args:
            dir_path: 目录路径.
            recursive: 是否递归子目录.
            metadata: 全局元数据.

        Returns:
            Document 列表.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            logger.warning("Directory not found: %s", dir_path)
            return []

        pattern = "**/*" if recursive else "*"
        all_docs = []

        for ext in self._handlers:
            for file_path in dir_path.glob(f"{pattern}{ext}"):
                try:
                    docs = self.load_file(str(file_path), metadata)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error("Failed to load %s: %s", file_path, e)

        logger.info(
            "Loaded %d docs from directory %s", len(all_docs), dir_path
        )
        return all_docs

    # ================================================================
    # 结构化数据加载
    # ================================================================

    def load_from_dicts(
        self,
        records: list[dict],
        content_key: str = "content",
        metadata_keys: Optional[list[str]] = None,
    ) -> list[Document]:
        """从字典列表构建文档.

        Args:
            records: 字典列表，每个字典至少包含 content_key.
            content_key: 用作文档内容的字段名.
            metadata_keys: 用作元数据的字段名 (默认: 所有非 content_key 字段).

        Returns:
            Document 列表.
        """
        docs = []
        for record in records:
            content = record.get(content_key, "")
            if not content:
                continue

            meta = {}
            if metadata_keys:
                meta = {k: record.get(k) for k in metadata_keys}
            else:
                meta = {
                    k: v for k, v in record.items()
                    if k != content_key and not isinstance(v, (list, dict))
                }

            docs.append(Document(content=str(content), metadata=meta))

        return docs

    def load_stock_profiles(
        self, stocks: list  # list[Stock]
    ) -> list[Document]:
        """从 Stock 对象构建结构化文档.

        每只股票生成一条结构化描述，适合向量检索。
        """
        docs = []
        for s in stocks:
            parts = [f"股票代码: {s.symbol}"]
            if s.name:
                parts.append(f"名称: {s.name}")
            parts.append(f"市场: {'A股' if s.market == 'CN' else '美股'}")
            if s.exchange:
                parts.append(f"交易所: {s.exchange}")
            if s.sector:
                parts.append(f"行业板块: {s.sector}")
            if s.industry:
                parts.append(f"细分行业: {s.industry}")
            parts.append(f"交易货币: {s.currency}")

            content = "；".join(parts)
            docs.append(Document(
                content=content,
                metadata={
                    "doc_type": "stock_profile",
                    "symbol": s.symbol,
                    "market": s.market,
                    "sector": s.sector or "",
                    "industry": s.industry or "",
                    "authority": 0.8,  # 基础信息权威度较高
                },
            ))

        return docs

    # ================================================================
    # 内部加载器
    # ================================================================

    def _load_text(self, path: Path) -> list[Document]:
        """加载纯文本/Markdown."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            return []
        return [Document(
            content=content,
            metadata={"source_file": str(path), "file_type": path.suffix},
        )]

    def _load_json(self, path: Path) -> list[Document]:
        """加载 JSON 文件.

        期望格式: {"documents": [{"content": "...", "metadata": {...}}, ...]}
        或 {"content": "...", "metadata": {...}}
        或 [{"content": "...", ...}, ...]
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return self.load_from_dicts(data)
        elif isinstance(data, dict):
            if "documents" in data:
                return self.load_from_dicts(data["documents"])
            elif "content" in data:
                return [Document(
                    content=data["content"],
                    metadata=data.get("metadata", {}),
                )]

        logger.warning("Unknown JSON structure in %s", path)
        return []

    def _load_jsonl(self, path: Path) -> list[Document]:
        """加载 JSONL 文件 (每行一个 JSON)."""
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "content" in record:
                        docs.append(Document(
                            content=record["content"],
                            metadata=record.get("metadata", {}),
                        ))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping invalid JSONL line: %s", e)
        return docs
