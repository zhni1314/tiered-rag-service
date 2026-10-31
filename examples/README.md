# 示例语料（sample corpus）

这里的 10 份 PDF 是**自编/整理的技术语料**，用来在没有真实文档的机器上跑通入库、问答与切块评测：

| 文件 | 内容 | 性质 |
|---|---|---|
| `ai-inference-platform-notes.pdf` | 虚构公司「星澜科技」推理平台的容量、灰度、回滚、事故与成本口径 | 完全虚构 |
| `rag-ops-handbook.pdf` | 虚构团队的检索系统运维制度：切块、重排、索引维护、评测门槛、故障处置 | 完全虚构 |
| `observability-notes.pdf` | 虚构的可观测性建设纪要：核心指标、日志、追踪、告警分级 | 完全虚构 |
| `release-engineering-notes.pdf` | 虚构的发布工程实践：主干开发、数据库变更、灰度、回滚、变更冻结 | 完全虚构 |
| `security-review-notes.pdf` | 虚构的 AI 应用安全评审清单：提示注入、脱敏、权限、工具白名单 | 完全虚构 |
| `ai-milestones-briefing.pdf` | 公开资料整理的技术里程碑与工程趋势摘要 | 公开事实摘要 |
| `inference-optimization-handbook.pdf` | 推理优化专题：两阶段延迟、KV 缓存、连续批处理、量化、投机解码、压测 | 技术专题 |
| `agent-engineering-notes.pdf` | 智能体工程专题：工具循环上限、权限边界、失败回退、评估方法 | 技术专题 |
| `multimodal-and-eval-notes.pdf` | 多模态输入与评测方法专题 | 技术专题 |
| `recent-announcements-digest.pdf` | 2026-10 官方公告摘录，只收录当天能从官方新闻页读到、且标注日期的条目 | 一手页面摘录 |

`.html` 是可编辑的源文件，`.pdf` 由它生成。重新生成（需要本机装有 Edge 或 Chrome）：

```powershell
& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --headless=old --disable-gpu `
  --no-pdf-header-footer "--print-to-pdf=ai-inference-platform-notes.pdf" `
  "file:///完整路径/ai-inference-platform-notes.html"
```

入库：

```bash
python scripts/build_index.py --dir examples/sample_corpus --level public
```

切块对比评测（flat 800 vs 父子 800/200，离线免费）：

```bash
python scripts/chunking_bench.py
```

题目与 gold 在 `scripts/chunking_dataset.json`：`sample-corpus-chunking-v2`，10 篇语料、60 题
（30 精确 / 15 同义 / 15 跨文档），gold 是语料中的唯一子串，`tests/test_chunking_dataset.py`
会在 CI 里校验唯一性和跨文档覆盖。v1（6 篇、25 题）的结果不与 v2 比较。

前五份里的公司名、阈值、排班都是编的，不要当成真实案例引用；
把它们写进文章或答辩时，请说明是演示语料。公告摘录只保留标题级信息，细节以官方页面为准。
