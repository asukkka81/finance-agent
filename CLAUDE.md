# CLAUDE.md

## 项目概述

**finance-agent** — 智能金融 DeepResearch Agent。
基于大模型智能体 (Qwen3-14B) 技术，整合全品类金融数据与专业投研知识，
打造面向个人客户、理财顾问、合规人员的智能咨询系统。

### 整体架构 (四层)

```
┌─────────────────────────────────────────┐
│  4. 模型层  Qwen 微调 + GRPO 对齐        │
├─────────────────────────────────────────┤
│  3. Agent层  MCP 协议调度 + 5大工具       │
├─────────────────────────────────────────┤
│  2. 检索层   向量DB + 多模态检索           │
├─────────────────────────────────────────┤
│  1. 数据层   yfinance + akshare + SQLite  │  ◄── 当前阶段
└─────────────────────────────────────────┘
```

### 当前阶段: 数据层

数据层是整个项目的基础设施，负责：
- 从 yfinance (美股) 和 akshare (A股/基金/宏观) 获取金融数据
- SQLite 本地持久化存储，UNIQUE 索引自动去重
- 提供标准化的查询接口，为后续 MCP tool handler 做准备

## 目录结构

```
finance-agent/
├── data_layer/                    # 数据层主包
│   ├── __init__.py                # 公共 API 导出
│   ├── config.py                  # YAML 配置加载器
│   ├── database/
│   │   ├── connection.py          # SQLite 连接管理 (WAL模式, 上下文管理器)
│   │   └── schema.py              # 7 张表的 DDL 定义
│   ├── models/                    # 数据传输对象 (dataclass)
│   │   ├── stock.py               # Stock
│   │   ├── price.py               # DailyPrice
│   │   ├── fund.py                # Fund, FundNAV
│   │   └── macro.py               # MacroIndicator, FinancialIndicator
│   ├── repositories/              # 数据库 CRUD 封装
│   │   ├── base.py                # BaseRepository 模板基类
│   │   ├── stock_repository.py    # 股票元数据 CRUD
│   │   ├── price_repository.py    # 行情 CRUD (核心)
│   │   ├── fund_repository.py     # 基金 CRUD
│   │   └── macro_repository.py    # 宏观 & 财务指标 CRUD
│   ├── fetchers/                  # 外部数据获取
│   │   ├── base.py                # AbstractFetcher 抽象类
│   │   ├── yfinance_fetcher.py    # 美股数据源
│   │   ├── akshare_fetcher.py     # A股/基金/宏观数据源
│   │   └── fetcher_factory.py     # Fetcher 工厂 (按市场/名称创建)
│   ├── services/                  # 业务逻辑编排
│   │   ├── data_service.py        # 核心服务: 同步 + 查询 + 搜索
│   │   └── sync_service.py        # 定时同步: 全量 / 增量
│   └── utils/
│       ├── date_utils.py          # 交易日历工具
│       └── validators.py          # 数据校验工具
├── retrieval_layer/               # 检索层主包
│   ├── __init__.py                # 公共 API 导出
│   ├── config.py                  # 检索配置 (模型/路径/参数)
│   ├── embeddings/                # Embedding 模型
│   │   ├── base.py                # BaseEmbedder 抽象类
│   │   ├── bge_embedder.py        # BGE 中文语义模型
│   │   └── factory.py             # Embedding 工厂 (单例缓存)
│   ├── stores/                    # 向量 & 全文存储
│   │   ├── base.py                # BaseVectorStore / BaseTextStore
│   │   ├── chroma_store.py        # ChromaDB 向量存储
│   │   ├── fts5_store.py          # SQLite FTS5 全文索引 (BM25 + CJK分词)
│   │   └── factory.py             # Store 工厂
│   ├── indexer/                   # 文档入索引流水线
│   │   ├── loader.py              # DocumentLoader (文件/Stock/Dict)
│   │   ├── splitter.py            # TextSplitter (中文友好递归分块)
│   │   └── indexer.py             # Indexer 编排器
│   ├── retrieval/                 # 检索策略
│   │   ├── dense.py               # DenseRetriever (向量语义)
│   │   ├── sparse.py              # SparseRetriever (BM25 关键词)
│   │   ├── hybrid.py              # HybridRetriever (RRF 融合)
│   │   └── reranker.py            # Reranker (BGE Cross-Encoder 精排)
│   ├── pipelines/                 # 检索流水线
│   │   ├── ingestion.py           # IngestionPipeline (数据摄入)
│   │   └── search.py              # SearchPipeline (统一搜索入口)
│   └── utils/
│       └── text_utils.py          # 文本工具 (query规范化/关键词提取)
├── config/
│   └── settings.yaml              # 全局配置
├── data/                          # 数据库文件存放
│   └── chroma_db/                 # ChromaDB 向量文件
├── tests/                         # pytest 测试
│   ├── conftest.py                # 共享 fixtures (内存DB)
│   ├── test_database/
│   ├── test_repositories/
│   └── test_services/
├── notebooks/                     # Jupyter 探索分析
├── pyproject.toml
├── requirements.txt
└── CLAUDE.md                      # 本文件
```

## 数据库表设计

| 表名 | 用途 | 唯一约束 | 关键字段 |
|------|------|---------|---------|
| `stocks` | 股票元数据 | `symbol` | symbol, market, exchange, sector |
| `daily_prices` | 日线OHLCV行情 | `(stock_id, trade_date)` | open, high, low, close, volume, adj_close |
| `funds` | 基金元数据 | `code` | code, fund_type, manager |
| `fund_nav` | 基金净值 | `(fund_id, nav_date)` | unit_nav, accumulated_nav, daily_return |
| `financial_indicators` | 财务指标 | `(stock_id, report_date, report_type)` | PE, PB, ROE, 营收增速等 |
| `macro_indicators` | 宏观指标 | `(indicator_name, pub_date)` | CPI, PMI, M2, GDP |
| `fetch_logs` | 同步日志 | — | data_type, status, records_count |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 初始化数据库
python -c "
from data_layer.config import load_config
from data_layer.database.connection import DatabaseManager

config = load_config()
db = DatabaseManager(config.db_path)
db.initialize_schema()
print('Database initialized:', config.db_path)
"

# 3. 首次全量同步 (美股+A股+基金+宏观)
python -c "
from data_layer.config import load_config
from data_layer.database.connection import DatabaseManager
from data_layer.services.sync_service import SyncService

config = load_config()
db = DatabaseManager(config.db_path)
db.initialize_schema()

sync = SyncService(db, config)
results = sync.full_sync(years=3)
"

# 4. 查询数据
python -c "
from datetime import date
from data_layer.config import load_config
from data_layer.database.connection import DatabaseManager
from data_layer.services.data_service import DataService

config = load_config()
db = DatabaseManager(config.db_path)
svc = DataService(db, config)

# 查行情
df = svc.get_price_history('AAPL', date(2024, 1, 1), date.today())
print(df.head())

# 搜股票
results = svc.search_stocks('茅台')
print(results)
"

# 5. 运行测试
pytest tests/ -v

# 6. 检索层 - 入索引 (将数据层内容导入检索库)
python -c "
from data_layer.config import load_config
from data_layer.database.connection import DatabaseManager
from retrieval_layer.config import RetrievalConfig
from retrieval_layer.embeddings.factory import EmbeddingFactory
from retrieval_layer.stores.factory import StoreFactory
from retrieval_layer.pipelines.ingestion import IngestionPipeline
from retrieval_layer.pipelines.search import SearchPipeline

# 初始化
data_config = load_config()
db = DatabaseManager(data_config.db_path)
ret_config = RetrievalConfig()

embedder = EmbeddingFactory(device='cpu').get()
stores = StoreFactory(ret_config)
vector_store = stores.create_vector_store()
text_store = stores.create_text_store()

# 摄入股票数据
pipeline = IngestionPipeline(embedder, vector_store, text_store, ret_config)
pipeline.ingest_stocks(db)

# 搜索
search = SearchPipeline(embedder, vector_store, text_store, ret_config)
results = search.search('贵州茅台 PE 估值', top_k=5)
for r in results:
    print(f\"[{r['match_type']}] score={r['score']:.4f}: {r['document'][:80]}...\")
"
```

## 检索层架构

### 三路召回 → RRF 融合 → Reranker 精排

```
Query ──┬── Route 1: Dense (BGE 向量) ── ChromaDB 语义相似度 ──┐
        ├── Route 2: Sparse (FTS5 BM25) ── SQLite 关键词匹配 ──┤
        └── Route 3: (预留) GME 多模态 ── 图文跨模态 ──────────┘
                                        │
                                        ▼
                                  RRF 融合
                                        │
                                        ▼
                              BGE Reranker 精排
                                        │
                                        ▼
                            时效降级 + 权威过滤 + 去重
                                        │
                                        ▼
                                  Top-K 结果
```

### 核心设计决策

#### CJK 预分词
SQLite FTS5 默认 tokenizer 将连续中文视为单个 token，导致搜索失败。
解决方案: 在索引和查询时对 CJK 文本做字符级预分词 (加空格分隔)，
使每个汉字成为独立 token，FTS5 的 BM25 和短语搜索均正常工作。

#### ChromaDB 选型
本地持久化、Python-native、支持元数据过滤、HNSW 索引。
相比 FAISS 的优势: 自带元数据管理，无需额外维护映射表。

#### RRF 融合
`score = Σ 1/(k + rank_i)`, k=60。
相比线性加权: 不需要对各路分数做归一化，对 rank 偏差鲁棒。

#### 检索结果标准格式
所有检索器统一返回:
```python
{
    'id': str,          # 文档唯一ID
    'document': str,    # 原文内容
    'score': float,     # 0~1 相似度 (越大越好)
    'metadata': dict,   # 元数据 (doc_type/symbol/authority...)
    'source': str,      # 'dense' | 'sparse' | 'hybrid' | 'hybrid_reranked'
    'match_type': str,  # 'both' | 'dense_only' | 'sparse_only'
}
```

## 核心设计决策

### 为什么不用 ORM?
数据层是纯 ETL 场景 — SQL 操作简单固定（批量 INSERT OR IGNORE、按日期范围查询）。
直接写 SQL 更轻量、可控、性能更好。后续检索层引入向量数据库时再评估 ORM。

### 去重策略
利用 SQLite 的 `UNIQUE INDEX` + `INSERT OR IGNORE` 在数据库层面保证幂等。
不需要应用层先查再插，一次 `executemany` 搞定。效率高且不会出错。

### DataFrame 作为数据交换格式
Fetcher → Service → Repository 之间统一用 `pd.DataFrame` 传递数据。
金融分析天然适合 DataFrame，且后续 Python 沙箱直接消费 DataFrame。

### 增量更新
`get_latest_trade_date(stock_id)` → 从次日开始拉取 → 只写新数据。
避免每次全量拉取，节省 API 调用和带宽。

### MCP 接口预留
`DataService` 的查询方法设计为无状态、JSON 友好的签名:
- `get_price_history(symbol, start, end) → DataFrame`
- `search_stocks(keyword) → list[dict]`
- `get_latest_prices(market) → DataFrame`

后续 Agent 层只需给每个方法加 `@mcp_tool` 装饰器即可暴露为 MCP tool。

## 编码约定

- **语言**: Python 3.11+，类型注解全覆盖
- **命名**: snake_case (文件/函数/变量), PascalCase (类)
- **数据模型**: dataclass (不用 Pydantic)
- **日志**: `logging.getLogger(__name__)`
- **测试**: pytest + 内存数据库 (`:memory:`), 不依赖外部 API
- **配置**: YAML → `Config` 类 (属性访问)

## 扩展点

1. **新增数据源**: 实现 `AbstractFetcher` → 在 `FetcherFactory` 注册
2. **新增表**: 在 `schema.py` 添加 DDL → 写 `Repository` → `DataService` 加方法
3. **MCP 集成**: `DataService` 方法 → `@mcp_tool` 装饰器 → Agent 可调用
4. **向量检索集成**: 后续阶段在 `data_layer/` 同级建 `retrieval_layer/`
