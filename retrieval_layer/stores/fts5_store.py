# retrieval_layer/stores/fts5_store.py
"""SQLite FTS5 全文检索引擎.

FTS5 是 SQLite 内置的全文搜索扩展，支持:
    - BM25 相关性排序
    - 前缀查询
    - 短语查询
    - 中英文混合检索 (通过 CJK 字符级预分词)

CJK 处理策略:
    FTS5 默认 tokenizer 将连续 CJK 字符视为单个 token，
    导致中文搜索失败。解决方案是在索引和查询时对 CJK 文本
    做字符级预分词 (加空格分隔)，使每个汉字成为独立 token，
    同时保留多字符短语搜索能力。
"""

import logging
import re
import sqlite3
from contextlib import contextmanager
from typing import Optional

from retrieval_layer.stores.base import BaseTextStore

logger = logging.getLogger(__name__)


# CJK Unicode 范围
_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F), # CJK Compatibility Ideographs Supplement
]


def _is_cjk(char: str) -> bool:
    """判断字符是否为 CJK 汉字."""
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _tokenize_for_fts(text: str) -> str:
    """对文本做 CJK 预分词处理.

    在 CJK 字符周围插入空格，使其被 FTS5 识别为独立 token。
    同时保留原始的非 CJK token (英文单词、数字等)。

    例: "贵州茅台2024年营收1741亿元"
      → "贵 州 茅 台 2024 年 营 收 1741 亿 元"
    """
    if not text:
        return ""

    result = []
    for char in text:
        if _is_cjk(char):
            result.append(f" {char} ")
        else:
            result.append(char)

    # 规范化空白
    return " ".join("".join(result).split())

from retrieval_layer.stores.base import BaseTextStore

logger = logging.getLogger(__name__)


class FTS5Index(BaseTextStore):
    """SQLite FTS5 全文索引.

    Usage::

        fts = FTS5Index("data/fts5.db")
        fts.add(["doc1", "doc2"], ["文本内容1", "文本内容2"])
        results = fts.search("ROE 市盈率", top_k=10)
    """

    def __init__(self, db_path: str = "data/fts5.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库: 创建文档表 + FTS5 虚拟表.

        策略:
            - documents 表: id=行标识, doc_id=业务唯一ID, content=原文, content_fts=索引文本
            - documents_fts: FTS5 content-sync 模式 (同步 content_fts 列)
            - 三触发器 (INSERT/UPDATE/DELETE) 保持 FTS5 索引与 documents 表同步
        """
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id      TEXT UNIQUE NOT NULL,  -- 业务唯一 ID
                    content     TEXT NOT NULL,          -- 原始内容
                    content_fts TEXT NOT NULL,          -- CJK 分词内容 (FTS 索引源)
                    metadata    TEXT                    -- JSON string
                );
            """)

            # FTS5 content-sync: 使用 documents.id 作为 rowid
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
                USING fts5(
                    content_fts,
                    content=documents,
                    content_rowid=id
                );
            """)

            # 触发器: 自动同步 FTS5 索引
            conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
                    INSERT INTO documents_fts(rowid, content_fts)
                    VALUES (new.id, new.content_fts);
                END;

                CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, content_fts)
                    VALUES ('delete', old.id, old.content_fts);
                END;

                CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, content_fts)
                    VALUES ('delete', old.id, old.content_fts);
                    INSERT INTO documents_fts(rowid, content_fts)
                    VALUES (new.id, new.content_fts);
                END;
            """)

            # 首次创建时重建索引 (捕获已存在的数据)
            conn.execute(
                "INSERT INTO documents_fts(documents_fts) VALUES('rebuild')"
            )

            conn.commit()

        logger.info("FTS5 index initialized: %s", self.db_path)

    @contextmanager
    def _get_conn(self) -> sqlite3.Connection:
        """获取 SQLite 连接上下文."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ================================================================
    # CRUD
    # ================================================================

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """批量添加文档.

        原文存入 content 列，CJK 分词版本存入 content_fts 列。
        FTS5 通过 content= 自动同步 content_fts 列。
        """
        if not ids:
            return

        import json

        with self._get_conn() as conn:
            for i, doc_id in enumerate(ids):
                original = documents[i]
                tokenized = _tokenize_for_fts(original)
                metadata_json = json.dumps(
                    metadatas[i] if metadatas else {}, ensure_ascii=False
                )

                conn.execute(
                    "INSERT OR REPLACE INTO documents "
                    "(doc_id, content, content_fts, metadata) VALUES (?, ?, ?, ?)",
                    (doc_id, original, tokenized, metadata_json),
                )

        logger.debug("FTS5: added %d documents", len(ids))

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """全文检索 (BM25 排序).

        对查询文本做 CJK 分词后匹配 content_fts 列，
        返回原始 content 列的内容。
        """
        import json

        # 对查询做 CJK 分词 (与索引时的处理一致)
        tokenized_query = _tokenize_for_fts(query)

        sql = """
            SELECT
                d.doc_id as id,
                d.content as document,
                d.metadata,
                bm25(documents_fts) as score
            FROM documents_fts
            JOIN documents d ON documents_fts.rowid = d.id
            WHERE documents_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """

        with self._get_conn() as conn:
            try:
                rows = conn.execute(sql, (tokenized_query, top_k)).fetchall()
            except sqlite3.OperationalError as e:
                logger.warning("FTS5 query error: %s (query: %s)", e, query)
                return []

        results = []
        for row in rows:
            metadata = {}
            try:
                if row["metadata"]:
                    metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                pass

            results.append({
                "id": row["id"],  # doc_id
                "document": row["document"],
                "score": float(row["score"]) if row["score"] is not None else 0.0,
                "metadata": metadata,
            })

        # BM25 分数越小越好，归一化为越大越好
        if results:
            max_score = max(r["score"] for r in results)
            if max_score > 0:
                for r in results:
                    r["score"] = 1.0 - (r["score"] / max_score)

        return results

    def delete(self, ids: list[str]) -> None:
        """按 doc_id 删除文档 (FTS5 通过 content= 自动同步)."""
        if ids:
            with self._get_conn() as conn:
                placeholders = ", ".join(["?"] * len(ids))
                conn.execute(
                    f"DELETE FROM documents WHERE doc_id IN ({placeholders})",
                    tuple(ids),
                )

    def count(self) -> int:
        """返回文档总数."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return row[0] if row else 0

    def clear(self) -> None:
        """清空所有文档 (FTS5 自动同步)."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM documents")
        logger.info("FTS5 index cleared")

    # ================================================================
    # 高级查询
    # ================================================================

    def phrase_search(self, phrase: str, top_k: int = 10) -> list[dict]:
        """精确短语搜索 (用双引号包裹)."""
        return self.search(f'"{phrase}"', top_k)

    def prefix_search(self, prefix: str, top_k: int = 10) -> list[dict]:
        """前缀搜索."""
        return self.search(f"{prefix}*", top_k)
