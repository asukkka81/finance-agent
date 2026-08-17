#!/usr/bin/env python3
"""构建 RAG 检索库 — 真实长文语料离线索引.

语料来源 (按权威度):
    1. DISC-FinLLM retrieval_part 参考材料 (~100 段) — 金融分析/市场解读长文
    2. DISC-FinLLM task_part 上下文 (~100 段) — 公司/行业研究式长文
    3. FinRpt 研报抽样 (默认 400 条) — 券商研报分析文本 (news/report_write_response)
    4. csprd 政策语料抽样 (默认 500 条) — 产业政策文本

写入: fts5.db (BM25) + chroma_db (向量), 幂等 (Chroma upsert + 内容哈希 id).
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

# 以 `python scripts/xxx.py` 方式运行时, 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/training/raw")
FINRPT_PATH = Path("data/research_reports/FinRpt.jsonl")
CSPRD_PATH = Path("data/research_reports/csprd_policy_corpus.json")


# ================================================================
# 语料抽取
# ================================================================

def extract_disc_retrieval_materials(limit: int | None = None) -> list[dict]:
    """DISC retrieval_part: 从 instruction 中抽取「参考材料 N:」段落."""
    records = []
    data = json.loads(Path(RAW_DIR, "retrieval_part.json").read_text())
    pattern = re.compile(r"参考材料\s*\d+\s*[:：]\s*")
    for item in data:
        inst = item.get("instruction", "")
        parts = pattern.split(inst)
        # parts[0] 是任务说明, parts[1:] 是各参考材料
        for text in parts[1:]:
            text = text.strip()
            if len(text) >= 80:  # 过滤过短片段
                records.append({
                    "content": text,
                    "doc_type": "research_material",
                    "source": "DISC-FinLLM",
                    "authority": 0.7,
                })
        if limit and len(records) >= limit:
            break
    return records


def extract_disc_task_contexts(limit: int | None = None) -> list[dict]:
    """DISC task_part: 从 instruction 中抽取「上下文:」段落."""
    records = []
    data = json.loads(Path(RAW_DIR, "task_part.json").read_text())
    for item in data:
        inst = item.get("instruction", "")
        m = re.search(
            r"上下文\s*[:：]+\s*(.*?)(?=\s*(?:选项|答案|问题)\s*[:：]|\Z)",
            inst, re.DOTALL,
        )
        text = m.group(1).strip() if m else ""
        if len(text) >= 80:
            records.append({
                "content": text,
                "doc_type": "company_context",
                "source": "DISC-FinLLM",
                "authority": 0.7,
            })
        if limit and len(records) >= limit:
            break
    return records


def sample_finrpt(n: int, seed: int) -> list[dict]:
    """FinRpt 抽样: 研报分析文本 (report_write_response)."""
    lines = FINRPT_PATH.read_text().splitlines()
    rng = random.Random(seed)
    picks = rng.sample(range(len(lines)), min(n, len(lines)))
    records = []
    for i in picks:
        try:
            d = json.loads(lines[i])
        except json.JSONDecodeError:
            continue
        text = (d.get("report_write_response") or "").strip()
        if len(text) < 100:
            continue
        records.append({
            "content": text,
            "doc_type": "research_report",
            "source": "FinRpt",
            "stock_code": str(d.get("stock_code", "")),
            "authority": 0.8,
        })
    return records


def sample_csprd(n: int, seed: int) -> list[dict]:
    """csprd 政策语料抽样."""
    rng = random.Random(seed)
    lines = CSPRD_PATH.read_text().splitlines()
    picks = rng.sample(range(len(lines)), min(n, len(lines)))
    records = []
    for i in picks:
        try:
            d = json.loads(lines[i])
        except json.JSONDecodeError:
            continue
        text = (d.get("text") or "").strip()
        if len(text) < 100:
            continue
        records.append({
            "content": text,
            "doc_type": "policy",
            "source": "csprd",
            "authority": 0.85,
        })
    return records


# ================================================================
# 主流程
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finrpt", type=int, default=400, help="FinRpt 抽样条数")
    parser.add_argument("--csprd", type=int, default=500, help="csprd 抽样条数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logger.info("抽取语料...")
    records = []
    records += extract_disc_retrieval_materials()
    records += extract_disc_task_contexts()
    records += sample_finrpt(args.finrpt, args.seed)
    records += sample_csprd(args.csprd, args.seed)
    logger.info("语料共 %d 篇", len(records))

    from retrieval_layer.config import RetrievalConfig
    from retrieval_layer.embeddings.factory import EmbeddingFactory
    from retrieval_layer.pipelines.ingestion import IngestionPipeline
    from retrieval_layer.stores.factory import StoreFactory

    ret_config = RetrievalConfig()
    embedder = EmbeddingFactory(device="cpu").get()
    stores = StoreFactory(ret_config)
    pipeline = IngestionPipeline(
        embedder, stores.create_vector_store(), stores.create_text_store(), ret_config,
    )

    logger.info("开始离线索引 (分块 512/重叠 64, BGE-small CPU)...")
    n_chunks = pipeline.ingest_custom(records, content_key="content")
    logger.info("索引完成: %d 个 chunk", n_chunks)
    print(json.dumps(pipeline.get_stats(), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
