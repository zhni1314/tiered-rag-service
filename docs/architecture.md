# 架构说明

## 运行时对象图

```
FastAPI app
 └── app.state
     ├── settings      Settings          # 启动时读取，之后不可变
     ├── redis         Redis             # 会话历史 + LangChain LLM cache
     ├── history       HistoryStore
     ├── vectorstore   Chroma            # 只存子块；持久化在 settings.chroma_dir
     ├── parents       ParentStore       # parent_id → 父块正文，JSON，原子替换
     ├── pool          RetrieverPool     # {level -> ContextualCompressionRetriever}
     └── chain         Runnable          # 条件重写 → 检索 → 拼上下文 → 生成
```

启动顺序在 `service.lifespan` 里，**顺序有硬约束**：

1. `settings.require_llm_credentials()` — 缺密钥立刻失败，不要等到第一次请求。
2. `configure_huggingface(settings)` — 写入 `HF_HOME` / `HF_ENDPOINT`。
3. **然后**才导入 `HuggingFaceEmbeddings` / `HuggingFaceCrossEncoder`。

`huggingface_hub` 在 import 时就把缓存路径算成模块常量，所以第 3 步不能提到文件头，
否则 `.env` 里的 `HF_HOME` 无效。这就是 `service.py` 把重量级导入放进 lifespan 的原因，
`bootstrap.py` 存在的全部理由也是这个。

## 模块契约

| 模块 | 输入 | 输出 | 允许依赖 |
|---|---|---|---|
| `config` | env dict / 进程环境 | frozen `Settings` | 仅标准库 + dotenv |
| `text_cleaning` | 原始文本 + 正则 | 清洗后文本 | 仅 `re` |
| `indexer` | PDF 路径、档位 | `Ingested{子块, 父块字典}` | pypdf、text-splitters |
| `parents` | JSON 路径 | `id → 父块正文`，可持久化 | 仅标准库 |
| `context` | 重排后的子块 + 父块字典 | Prompt 上下文文本 | langchain-core |
| `retrieval` | `Document` 全集 | 按档编译的检索引擎 | langchain-community/classic |
| `chain` | LLM + pool + settings + parents | `Runnable` | langchain-core |
| `history` | Redis client | 提示文本 / 历史列表 | redis |
| `service` | 以上全部 | HTTP 行为 | fastapi |

依赖是单向的：`service` → 领域模块 → 三方库。反向依赖（领域模块 import `service`）不存在，
所以换 Web 框架不动检索逻辑，换检索策略不动 HTTP 层。

## RetrieverPool 的并发模型

`rebuild()` 在锁外构建完整引擎字典，成功后一次性替换引用；`get()` 只读字典。
上传路径经 `run_in_threadpool` 调 `pool.replace_source()`，其间可能有 `/chat` 正在 `get()`：
读者要么拿到旧快照，要么拿到新快照，不会读到构建中的半成品。`replace_source` 先按
`source` 丢掉旧子块再重建 BM25，否则重传同名文件会让同一段文本在候选集里出现两份。

代价是上传期间到达的请求仍用旧索引 —— 这是可接受的最终一致性，比加读写锁复杂度划算。

## 数据流：一次 `/chat`

```
1. HistoryStore.as_prompt_text(session_id)      # Redis LRANGE，取最近 window 条
2. chain.ainvoke({question, history, user_level})
   2.1 smart_rewrite  # 无历史 or 无指代词 → 直接返回原问题（省一次 LLM 调用）
   2.2 pool.get(user_level) → engine
   2.3 asyncio.to_thread(engine.invoke, query)
       2.3.1 BM25 top-20（该档专属语料）
       2.3.2 Chroma top-20 + metadata filter {level: {$in: allowed}}
       2.3.3 EnsembleRetriever 融合去重
       2.3.4 CrossEncoder 打分 → top-10 子块
   2.4 context.assemble_context：按名次去重父块 → 3 块 / 3200 字预算
   2.5 Grounding Prompt 填充 → LLM → StrOutputParser
3. HistoryStore.append_turn(...)                # RPUSH + LTRIM 保持有界
4. ChatResponse{answer, latency, level}
```

第 2.3.1 和 2.3.2 是**冗余的**：BM25 已经只吃该档语料，metadata filter 又过滤一次。
双保险是有意的 —— 单靠内存 BM25 会随进程状态出错，向量库的 filter 是不可绕过的边界。
