# 面试准备手册 — 智能金融投研 Agent（DeepResearch）

> 本手册按简历条目组织：每条 = 简历原文 → 真实实现 → 代码位置 → 面试问答。
> 所有数字均可回溯到项目实际代码与数据文件（`data/test_logs/` 下有 10 问评测全轨迹）。

---

## 0. 项目全景（3 分钟版本）

**一句话**：面向个人客户的智能投研咨询系统，数据层（多源行情 + SQLite）→ 检索层（混合检索）→ Agent 层（MCP 工具调度 + 自评闭环）→ 模型层（SFT/GRPO 管线设计）。

```
用户问题
  │
  ▼
Agent 层: 意图解析(规则) ──► ReAct 循环 (LLM ⇄ 8+ MCP 工具, 最多6轮)
  │                              │
  │                    ┌─────────┼──────────┐
  │                    ▼         ▼          ▼
  │              数据工具   检索工具(RAG)  沙箱/SQL
  │              (行情查询) (BGE+BM25+RRF) (计算/统计)
  │                    └─────────┼──────────┘
  ▼                              ▼
最终答案 ──► 自评-修正闭环: 四维校验 + LLM judge
  │          低分(<0.7) → 带问题反馈重生成 (最多2次) → 仍差则降级标注
  ▼
返回用户 (含合规免责声明)
```

**关键数字**（全部实测）：8+ MCP 工具 ｜ 1100+ 检索文档 ｜ 7240 行 A 股行情（3 年）｜ 401 条真实验证 Text2SQL 样本 ｜ 单问处理 13s~3.5min ｜ 10 场景评测。

---

## 1. 简历条目一：MCP 协议 + ReAct 式多轮工具调用循环

### 真实实现

- MCP 工具注册：`agent_layer/mcp/registry.py`（`register` 装饰器 → 名称/描述/JSON Schema/角色注册）
- ReAct 循环：`agent_layer/orchestrator.py` 的 `_run_with_llm()`：
  ```
  第1轮: chat_with_tools(query + 工具 schema) → LLM 返回 tool_calls
  执行所有 tool_calls → 结果回传 chat_with_tool_results
  重复直到 LLM 返回纯文本答案, 上限 max_tool_rounds=6
  ```
- 工具集：行情查询 get_stock_price / 股票搜索 search_stocks / 宏观指标 get_macro_indicator / 知识检索 search_knowledge / 股票档案 search_stock_info / SQL 查询 execute_sql / 沙箱计算 execute_python
- 评测证据：Q5"对比茅台五粮液"两次工具调用完成、Q2/Q7 跑出 3 步以上依赖链（行情→SQL→沙箱）

### 核心代码（简化）

```python
# orchestrator.py — ReAct 主循环
for round_num in range(1, max_rounds + 1):
    if round_num == 1:
        resp = self.llm_client.chat_with_tools(messages, tools=tools, system_prompt=...)
    else:
        resp = self.llm_client.chat_with_tool_results(messages, tool_results=last_results, ...)
    if resp.has_tool_calls:
        for tc in resp.tool_calls:
            result = self.registry.execute(ToolCall(...))   # MCP 调度
            round_results.append({... result.data ...})
        messages.append(assistant tool_calls 消息)
        continue
    break  # 纯文本 → 最终答案
```

### 面试问答

- **Q: 为什么用 MCP 协议而不是直接函数调用？**
  A: ① 工具接口标准化（统一 JSON Schema 描述），LLM 工具选择与执行解耦；② 工具可插拔——数据/检索/沙箱/实时工具都是运行时注册，新增数据源只需 `@registry.register`；③ 与业界 Agent 生态对齐（Claude/各类 Agent 框架同协议）。
- **Q: ReAct 循环会不会死循环？**
  A: 三重防护——`max_tool_rounds=6` 硬上限；超轮数强制要求 LLM 基于已有结果作答；沙箱 30s 超时、SQL 只读限制。评测中最多跑满 6 轮后强制收尾。
- **Q: 多跳推理怎么体现？**
  A: 工具调用间存在数据依赖的链式调用。如"计算茅台波动率"：get_stock_price 拿价格 → execute_sql 补全数据 → execute_python 计算，3 步依赖链由 LLM 自主规划。

---

## 2. 简历条目二：多工具协同——混合检索（BGE + BM25 + RRF + Reranker）

### 真实实现

- 架构：`retrieval_layer/` 三路召回 → RRF 融合 → 精排
  - Dense 路：BGE-small-zh-v1.5（512 维）→ ChromaDB（HNSW 索引）
  - Sparse 路：SQLite FTS5 + BM25
  - RRF 融合：`score = Σ 1/(k + rank_i)`，k=60（`retrieval/hybrid.py`）
  - 精排：BGE-reranker-base 交叉编码器（`retrieval/reranker.py`）
- **CJK 预分词（简历里的"中文分词适配"）**：FTS5 默认分词器把连续中文当**一个 token**，"茅台"搜不到"贵州茅台"。方案：索引和查询两端对汉字做字符级插空格预分词，每字一个 token（`fts5_store.py` 的 `_tokenize_for_fts`）
- 语料：1100+ 文档 = DISC 研报材料 290 + FinRpt 研报 400 + csprd 政策 408 + 手写概念 49 + 股票档案 14（`scripts/build_rag_index.py` 可重建，幂等）

### 核心代码（简化）

```python
# hybrid.py — RRF 融合
def _rrf_fusion(self, dense_results, sparse_results, k=60):
    scores = {}
    for rank, r in enumerate(dense_results, 1):
        scores[r['id']] = scores.get(r['id'], 0) + 1/(k + rank)
    for rank, r in enumerate(sparse_results, 1):
        scores[r['id']] = scores.get(r['id'], 0) + 1/(k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### 面试问答

- **Q: 为什么不用纯向量检索？**
  A: 金融领域强术语依赖（"600519"、PE/ROE、专有名词），纯语义容易漏召回；BM25 关键词精确匹配补上术语类查询。评测里"夏普比率"类查询两路都命中（match_type=both）。
- **Q: RRF 为什么比加权融合好？**
  A: 两路分数分布不可比（余弦相似度 vs BM25 负对数分），加权需要归一化且对异常敏感；RRF 只用排名，对分数量纲天然鲁棒。
- **Q: 中文检索踩了什么坑？**
  A: FTS5 默认 tokenizer 对中文无效（整句一个 token）→ BM25 完全失效。做了字符级预分词，索引/查询两端对称处理，保证短语和关键词搜索可用。这是中文全文检索的经典坑。

---

## 3. 简历条目三：SQL 数据库查询 + Text2SQL 训练数据

### 真实实现

- `execute_sql` 工具：`agent_layer/tools/data_tools.py` 的 `register_sql_tool`——只读白名单（仅 SELECT/WITH）、100 行上限、错误回传
- 数据底座：SQLite 7 表（stocks / daily_prices / funds / fund_nav / financial_indicators / macro_indicators / fetch_logs），UNIQUE 索引 + INSERT OR IGNORE 幂等写入
- **401 条 Text2SQL 训练样本**：`scripts/rebuild_text2sql.py`——12 类问题模板（区间涨跌幅/年度排行/均线/标准差/对比…），**每条 SQL 真实执行验证**，答案来自执行结果（区别于旧版"问题与库错配 + SQL 未执行 + 占位答案"）

### 面试问答

- **Q: Text2SQL 训练数据怎么保证质量？**
  A: 三点：① 问题由本地库真实 schema 生成（不套用外部数据集的问题）；② 每条 SQL 生成后立即执行，执行失败或空结果直接丢弃；③ 答案文本由执行结果程序化生成，杜绝占位/编造。最终 401 条 100% 通过复验。
- **Q: SQL 工具怎么防注入/误操作？**
  A: 只读文件句柄打开数据库（`file:...?mode=ro`）+ 语句白名单 + 行数上限。

---

## 4. 简历条目四：Python 沙箱执行引擎

### 真实实现

- 架构：AST 静态扫描（`sandbox/scanner.py`）→ 受限 builtins 子进程执行（`sandbox/executor.py`）→ 30s 超时 + 临时文件清理
- 预置金融函数：sharpe_ratio / max_drawdown / rsi / macd / beta / efficient_frontier / value_at_risk / plot_* 等（`sandbox/runtime.py`）
- 安全机制：黑名单 import（os/sys/subprocess/socket…）+ 危险函数拦截（eval/exec/open）+ 白名单 import（numpy/pandas/math）

### 面试问答

- **Q: 沙箱怎么保证安全？**
  A: 三层——AST 静态扫描拦截危险 import 和函数调用；受限 builtins 的子进程隔离（不继承宿主命名空间）；超时与资源限制。**诚实口径**：这是演示级安全，AST 黑名单存在绕过路径（如 `__builtins__` 下标调用），生产环境应上容器级隔离（Docker/gVisor），这点评测复盘里已识别并列入修复计划。
- **Q: 评测暴露了沙箱什么问题？**
  A: 最大的工程发现是"数据桥接"——LLM 反复假设沙箱能访问前序工具结果（`data` 变量），实际沙箱完全隔离导致 NameError 频发。已列入修复计划：沙箱注入最近工具结果或明确工具语义。

---

## 5. 简历条目五：LLM 自评-修正闭环（self-refinement）

### 真实实现（本项目的核心工作量）

- 四维校验 `agent_layer/core/verifier.py`：
  - completeness：实体覆盖（提取查询中的股票代码/名称逐项核对）+ 需求清单（估值/行业/对比/投资等 8 类需求关键词检查）+ 数据类型覆盖
  - accuracy / logic：**LLM judge**——把「查询+答案+工具结果」发给大模型评审，返回 JSON 分数；解析失败三级回退到规则校验
  - compliance：风险提示关键词 + 买卖建议红线
- 闭环 `orchestrator.py` 的 `_verify_and_regenerate()`：
  ```
  校验分数 < 0.7 → 把 issues 拼进 REGENERATION_PROMPT (含原答案+工具数据)
  → LLM 重写 → 重新校验 → 有改善且仍低分则再重写 (最多2次)
  → 无改善提前停止 → 仍不合格降级: 追加免责声明+质量提示
  ```
- **评测真实战果**（可对面试官讲的案例）：
  - Q7"宁德最大回撤"：模型编造低点 136.82（真实最低 130.70），judge 三维全 FAIL，触发重答 2 次后降级标注——**校验系统成功识别数据编造**
  - Q1"最近30交易日"：模型误用 2024 年日期，judge 在 issues 中指出日期与工具结果不一致——**日期错位被捕获**
  - 诚实补充：Q6 的模拟数据编造被 judge 放行（模型把编造数据写进沙箱执行产生自洽假结果），accuracy<0.7 强制重答的加固已列入修复

### 核心代码（简化）

```python
# verifier.py — judge 调用
def _llm_judge(self, query, answer, tool_results):
    prompt = VERIFICATION_JUDGE_PROMPT.format(query=..., answer=..., tool_results=...)
    resp = self.llm_client.chat([{"role": "user", "content": prompt}], ...)
    return self._parse_judge_json(resp.content)  # 解析失败 → 规则回退

# orchestrator.py — 低分重答循环
for attempt in range(1, config.max_regeneration_rounds + 1):
    if best.overall_score >= config.min_verification_score:  # 默认 0.7
        break
    feedback = REGENERATION_PROMPT.format(issues=...)
    candidate = llm_client.chat(feedback).content
    new_v = verifier.verify_final_answer(candidate)
    if new_v.overall_score > best.overall_score:
        best = new_v          # 只保留更优答案
    else:
        break                  # 无改善提前止损
```

### 面试问答

- **Q: 为什么不直接用 LLM 生成完就返回？**
  A: 评测显示 LLM 在计算链路失败后会"假装算出来了"（编造 241 个价格、虚构标准差 1.2876%），金融场景答案错误代价高，所以加了独立质检阶段。
- **Q: 自评会不会误伤好答案 / 放行坏答案？**
  A: 实测 judge 抓假能力较强（能发现 21≠20、日期矛盾、虚构数字）；但存在放行案例。当前加固方向：accuracy < 0.7 视为 FAIL 级强制重答；重答 prompt 逐条引用 judge 指控。
- **Q: 这个方案和业界什么技术相关？**
  A: 属于 self-refinement / LLM-as-judge 路线（Self-Refine、Reflexion 同源思想），工程上结合了金融合规规则校验（关键词红线）。

---

## 6. 简历条目六：Qwen3-14B 基座 + LoRA SFT + GRPO 两阶段对齐 + 四维奖励函数

### 真实实现

- 训练管线代码齐全（`model_layer/`）：数据格式化（Qwen chat template）、SFT（PEFT LoRA + bitsandbytes）、GRPO（TRL 优先/自研简化循环回退）、评估（metrics/benchmark）、推理（vLLM 优先）
- **四维奖励函数** `model_layer/grpo/reward.py`：
  1. 格式合规（风险提示词、免责声明）
  2. 工具调度（调用工具名是否合法/合理）
  3. 代码质量（危险 import/函数模式检测——复用沙箱扫描器）
  4. 答案质量（长度/结构启发式）
- 训练数据：DISC-FinLLM 357 条（复旦专家数据集：咨询/计算/检索/任务四类）+ Text2SQL 401 条；数据分工设计：SFT 用 DISC（模仿语料），GRPO 用博金 1000 问 + 官方库验证（客观奖励），评测用 holdout——**三集不重叠**
- **诚实口径（面试必须准备）**：管线与数据就绪，真实训练尚未执行（需 A100 级 GPU）；奖励函数当前是字符串启发式，规划接入真实 Agent 执行结果作为奖励

### 面试问答

- **Q: 为什么 SFT 和 GRPO 数据不能重复？**
  A: SFT 是模仿学习（见过标准答案），GRPO 是探索-奖励（从没见过的 prompt 出发采样打分）。同源会让 rollout 缺乏多样性、奖励无信息量（模型已背过答案）、过拟合评测集。所以 SFT 用 DISC、GRPO 用博金、评测用 holdout 三段隔离。
- **Q: 奖励函数为什么是四维？**
  A: 对齐"好答案"的四个维度：合规（金融红线）、工具调度（调用正确工具）、代码质量（沙箱可执行且安全）、答案质量（完整有据）。合成奖励加权（权重和≈1）。
- **Q: GRPO 相比 PPO 的优势？**
  A: 无需价值模型（用组内相对比较计算 advantage），显存与训练开销更低，适合 14B 级模型的 RLHF 对齐。

---

## 7. 简历条目七（成效）：分钟级响应 + 多跳推理

### 数字口径（全部可回溯）

| 声明 | 实测证据 |
|---|---|
| 数十分钟 → 分钟级 | 10 问评测：最快 13.5s（知识问答），最慢 211s（多跳计算），均值约 2 分钟；对比人工投顾咨询数十分钟起 |
| 多跳推理 | Q2 波动率：行情→SQL→沙箱 3 步依赖链；Q6：行情→沙箱→行情→沙箱 4 步 |
| 全链路基建 | 7240 行行情（10 只 A 股×3 年，新浪源前复权）；1100+ 检索文档；401 条验证样本 |

### 面试问答

- **Q: 耗时瓶颈在哪？**
  A: LLM 轮次（每轮一次 API 调用）+ 失败重试。复杂计算题 200s+ 主要是工具调用失败后的纠正轮次（如沙箱 NameError 重试）。优化方向：沙箱数据注入（省掉纠正轮）、工具并行执行。

---

## 8. 简历口径风险点（面试前必看）

| 简历表述 | 实际状态 | 建议口径 |
|---|---|---|
| 覆盖"持仓分析" | ❌ 无持仓数据/功能（无用户组合表） | "系统设计目标包含持仓分析，当前版本聚焦行情/知识/计算，持仓模块在规划中" |
| "风险评估" | △ 合规校验的风险提示 + 需求清单风险维度，无完整风险评估模型 | "风险评估通过合规校验和风险提示实现，覆盖'不构成投资建议'等红线" |
| "降低内容幻觉" | 设计目标；评测显示 judge 能识别大部分但仍有放行 | "自评闭环显著提升幻觉识别率，10 场景评测中捕获数据编造/日期错位等案例；仍存在改进空间（已列修复计划）" |
| "通过 LoRA SFT+GRPO 两阶段对齐优化" | 管线+数据就绪，训练未执行 | "完成两阶段对齐管线设计与训练数据构建（DISC 357 + Text2SQL 401），真实训练待 GPU 资源" |
| 沙箱"安全执行" | 演示级（AST 黑名单有绕过路径） | 见第 4 节诚实口径 |

---

## 9. 高频追问清单（快速自测）

1. 为什么选 SQLite 不用 MySQL？→ 单机部署、ETL 场景固定 SQL、WAL 并发够用；向量库才引入 ChromaDB
2. 数据幂等怎么保证？→ UNIQUE 索引 + INSERT OR IGNORE，不先查后插
3. 数据从哪来？→ akshare（新浪/东财接口），东财限流后实现多源降级
4. 增量同步怎么做？→ 按股票取 MAX(trade_date)+1 起拉，只写新增
5. 检索结果怎么避免无关文档？→ RRF 融合 + Reranker 精排 + authority 过滤 + 100 字符前缀去重
6. ChromaDB 为什么不用 FAISS？→ 自带元数据管理，无需维护外部映射表
7. 评测怎么设计的？→ 10 类真实问题（行情/计算/知识/SQL/对比/统计/回撤/行业/混合/均线），全轨迹落盘（意图/每轮决策/工具结果/校验/答案），答案与库内真值比对
8. 评测发现了什么问题？→ 沙箱数据桥接缺失、失败后编造数据、日期锚定缺失、judge 放行、提示词与工具集不一致——全部有轨迹证据
9. 最大的技术难点？→ ① 校验闭环设计（judge 解析容错/重答止损/降级兜底）② 中文 FTS5 分词适配 ③ 评测驱动的问题定位（数据编造）
10. 如果重做会改什么？→ 沙箱注入机制先行；答案"拒绝编造"的 prompt 约束；accuracy 硬门槛

---

## 10. 代码地图（被追问时快速定位）

| 想看什么 | 去哪找 |
|---|---|
| ReAct 主循环 / 重答闭环 | `agent_layer/orchestrator.py` |
| 四维校验 / LLM judge | `agent_layer/core/verifier.py` |
| MCP 工具注册与执行 | `agent_layer/mcp/registry.py` |
| 数据工具 / execute_sql | `agent_layer/tools/data_tools.py` |
| 混合检索 / RRF | `retrieval_layer/retrieval/hybrid.py` |
| FTS5 CJK 分词 | `retrieval_layer/stores/fts5_store.py` |
| 沙箱扫描器 / 执行器 | `sandbox/scanner.py` / `sandbox/executor.py` |
| 四维奖励函数 | `model_layer/grpo/reward.py` |
| SFT / GRPO 训练器 | `model_layer/sft/trainer.py` / `model_layer/grpo/grpo_trainer.py` |
| Text2SQL 数据生成 | `scripts/rebuild_text2sql.py` |
| RAG 索引构建 | `scripts/build_rag_index.py` |
| 10 问评测 harness / 全轨迹日志 | `scripts/run_agent_eval.py` / `data/test_logs/` |
| 数据层 schema / 同步服务 | `data_layer/database/schema.py` / `data_layer/services/sync_service.py` |
