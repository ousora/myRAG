# myRAG — TODO

## ✅ 已完成：Ingest 管线

```
.doc/file → parse → clean → format → chunk → embed → sqlite-vec
```

| 模块 | 状态 |
|------|------|
| 多格式解析 (PDF/DOCX/HTML/MD/TXT) | ✅ MarkItDown + Trafilatura |
| 文本清洗 (control chars, page breaks, whitespace) | ✅ TextCleaner + YAML rules |
| LLM 结构化 (title, tags, sections) | ✅ format_text_async() |
| Markdown 渲染 + 输出 | ✅ write_to_md() / format_md() |
| LangChain chunking (header-aware + oversized split + plain-text fallback) | ✅ |
| bge-m3 嵌入 | ✅ Embedder |
| sqlite-vec 持久化 (chunks + doc + FTS5) | ✅ process_file_hybrid(store_path=...) |
| 配置集中管理 | ✅ conf/config.yaml + config.py |
| 端到端验证 (cncc.txt) | ✅ 20 chunks, 向量检索准确 |
| 单元测试 | ✅ 22 passed |

---

## 🚧 缺口：Query 端

ingest 把数据存进去了，但没有取出来的统一接口。

| 缺口 | 说明 |
|------|------|
| **RAG 查询函数** | `rag_query(question, db_path)` — 检索 → 拼 context → 调 LLM 生成答案 |
| **CLI search 命令** | `python -m myrag.pipeline search "question" --db data/cncc.db` |
| **重新索引** | 同一 doc_id 重复 ingest 时的去重/更新策略 |
| **批量摄入 + 存储** | `process_directory()` 目前不走 sqlite-vec |

---

## 📋 待办（按优先级）

1. **[P0] RAG 查询接口** — 闭环：用户提问 → 检索 → 生成答案
2. **[P1] CLI search 子命令** — 终端直接查
3. **[P2] 批量摄入进 sqlite-vec** — process_directory 接入 storage
4. **[P3] 去重/更新** — 同一 doc_id 重复索引的处理
