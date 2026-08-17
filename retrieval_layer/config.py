# retrieval_layer/config.py
"""检索层配置."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RetrievalConfig:
    """检索层全局配置.

    Attributes:
        chroma_path: ChromaDB 持久化路径.
        fts5_db_path: SQLite FTS5 数据库路径.
        embedding_model: BGE embedding 模型名称.
        reranker_model: 重排序模型名称.
        chunk_size: 文本分块大小 (tokens).
        chunk_overlap: 分块重叠大小.
        top_k_dense: 向量召回数量.
        top_k_sparse: 全文召回数量.
        top_k_final: 最终返回数量.
        enable_reranker: 是否启用重排序.
    """

    # --- 存储路径 ---
    chroma_path: str = "data/chroma_db"
    fts5_db_path: str = "data/fts5.db"

    # --- 模型 ---
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"  # cpu | cuda | mps
    embedding_batch_size: int = 32
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- 分块 ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- 检索参数 ---
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 10
    enable_reranker: bool = True

    # --- 融合参数 ---
    rrf_k: int = 60  # RRF 融合常数

    # --- 过滤 ---
    max_age_days: Optional[int] = None  # 时效降级 (None = 不限)
    min_authority_score: float = 0.0    # 最低权威分

    def __post_init__(self):
        # 确保存储路径存在
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        Path(self.fts5_db_path).parent.mkdir(parents=True, exist_ok=True)
