# myRAG Pipeline 优化计划

## Completed (Phase 5 + Phase 6)

| # | Issue | Status | Resolution |
|---|-------|--------|------------|
| 1 | `validate_format_output()` / `try_fix_common_issues()` dead code | **WONTFIX** | Functions are used by `_format_text_single()` and `_format_text_chunked()` — not dead, just called only after LLM returns. Verified via call chain in formatters/__init__.py:619-622, 745-748. |
| 3 | `_executor` ThreadPoolExecutor never shutdown | **FIXED** | `atexit.register(_shutdown_executor)` added (phase 5). |
| 4 | FTS5 no sync mechanism | **FIXED** | Content-synced triggers (`chunks_ai/au/ad`) added to schema setup. |
| 6 | `call_llm()` unconditionally writes tmp/raw/ | **FIXED** | Gated by `debug_log_llm_responses` config flag (disabled by default). |
| 8 | Embedder `__new__()` bypasses normal construction | **FIXED** | Factory function `create_embedder()` added with proper Union typing. |
| 9 | Section filter LIKE wildcard injection | **FIXED** | `_escape_like_pattern()` escapes `\`, `%`, `_`. Also fixed underlying false-positive by using `json_each()` unnest (Phase 6). |
| 11 | Embedder no dimension validation | **FIXED** | `_validate_embedding_dimension()` raises `EmbeddingError` on mismatch. |
| 13 | CJK text chunk threshold inaccurate | **FIXED** | `effective_chunk_threshold()` with CJK-aware ratio detection (Phase 6). |
| 15 | Sentence split at abbreviations like "Mr." | **FIXED** | `_SENTENCE_ABBREVIATIONS` set + `_is_abbreviation_boundary()`. |
| 16 | O(n*m) heading lookup in chunker | **FIXED** | Pre-built `heading_by_line` dict for O(1) access. |
| 17 | Dead code after return in `_split_by_sentence` | **FIXED** | Removed unreachable merge-back-to-chunk-size block. |
| 18 | `_load_sqlite_vec()` fragile install detection | **FIXED** | Strategy 1: `importlib.import_module()`. Strategy 2: filesystem fallback. |

## P0 — Bug 修复（阻塞性）

| # | 问题 | 位置 | 说明 |
|---|------|------|------|
| 2 | `conftest.py` 含未清理的 `test_debug_import()` — pytest 会当作测试运行 | `storage/tests/conftest.py:3` | 应删除或改名为非 test_ 前缀 |

## P1 — 健壮性提升（高优先级）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 7 | JSON 提取正则 `\{.*\}` with DOTALL 会捕获多 {...} 间的垃圾文本 | `formatters/__init__.py:165` | LLM 输出含解释性文字时解析失败；建议用 json_repair 库替代 |
| 10 | Chunker `chunk()` 的 `section_path` 参数标注"Ignored"但保留在签名中 | `chunkers/__init__.py:70` | API 误导，应移除或真正支持覆盖 |

## P2 — 代码质量（中优先级）

| # | 问题 | 位置 | 建议 |
|---|------|------|------|
| 12 | TextCleaner / Chunker facade 类只覆写了部分特性 | `pipeline/core.py:114-142` | TextCleaner facade 测试缺失；应验证与 canonical impl 行为一致 |
| 14 | _extract_tags_from_body() 硬编码英文停用词 + 英文国家名集合 | `formatters/__init__.py:321-434` | 中文文档标签提取完全失效；需多语言停用词表或接入 LLM 生成标签 |

## P3 — 测试覆盖补全（重要）

| 模块 | 缺失内容 | 优先级 |
|------|----------|--------|
| formatters | _preprocess_json() 多场景参数化测试 | High |
| formatters | chunked processing path 端到端 mock 测试 | High |
| formatters | wikilink insertion + protected range 测试 | Medium |
| storage | FTS5 full-text search 功能测试（当前完全缺失） | High |
| storage | hybrid_search RRF 排序验证 | Medium |
| chunkers | plain text fallback path（无 markdown header）测试 | High |
| parsers | dispatcher 空扩展名 / .tar.gz 双后缀处理 | Low |
| writers | YAML front matter + metadata block 输出格式正确性 | Medium |

## 架构优化建议（Remaining）

1. ~~统一异常体系~~ — Phase 5 已实现 (`src/myrag/exceptions.py`)。
2. ~~配置验证接入启动流程~~ — Phase 5 `get_config()` 已调用 `_validate()` 并在失败时抛出自描述错误。
3. ~~Hybrid search 性能优化~~ — Phase 6 已将循环内重复的 `serialize_float32` + 单行查询改为一次序列化 + `IN (...)` 批量查询。
4. ~~Chunker 重叠逻辑改进~~ — Phase 5 `_apply_overlap()` 已扩展至最近空格边界，避免截断在单词中间。
5. **清理 `get_config_lazy()` 冗余** — 与 `get_config()` 功能完全相同，仅一处调用且无 circular import 问题，可直接替换为 `get_config`。

## 执行建议（Remaining）

```
Phase 7 (P0, ~30min): 删除 conftest.py 中的 test_debug_import()
Phase 8 (P1, ~2h): JSON resilience (json_repair) + Chunker API cleanup
Phase 9 (P2, ~4h): Tag extraction i18n for CJK docs
Phase 10 (测试, ~3h): 补全 formatter/storage/chunker 关键路径覆盖
```

总计约 **10 小时**工作量。
