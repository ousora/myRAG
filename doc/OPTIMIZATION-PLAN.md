# myRAG Pipeline 优化计划

## P0 — Bug 修复（阻塞性）

| # | 问题 | 位置 | 说明 |
|---|------|------|------|
| 1 | `validate_format_output()` / `try_fix_common_issues()` 是死代码，从未被调用 | `formatters/prompts.py` | 两个完整函数（~50行）写了但没接入流程 |
| 2 | `conftest.py` 含未清理的 `test_debug_import()` — pytest 会当作测试运行 | `storage/tests/conftest.py:3` | 应删除或改名为非 test_ 前缀 |
| 3 | `_executor` ThreadPoolExecutor 永不 shutdown，CLI 场景泄漏线程 | `formatters/__init__.py:40-48` | 应注册 atexit.register() 或在上下文管理器中清理 |
| 4 | FTS5 虚拟表无内容同步机制 — chunk 更新/删除后索引过期 | `storage/sqlite_vec.py:100,129` | 需添加 REINDEX trigger 或手动 sync |
| 5 | `_fix_bare_quotes_in_body_field()` 零测试覆盖，逻辑极脆弱 | `formatters/__init__.py:181-229` | 字符级 JSON 解析，任何结构变化都会静默失败 |

## P1 — 健壮性提升（高优先级）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 6 | `call_llm()` 无条件写 tmp/raw/，无限增长磁盘 | `formatters/__init__.py:124-131` | 生产环境应开关可控（config flag），非 debug 时不写盘 |
| 7 | JSON 提取正则 `\{.*\}` with DOTALL 会捕获多 {...} 间的垃圾文本 | `formatters/__init__.py:165` | LLM 输出含解释性文字时解析失败；建议用 json_repair 库替代 |
| 8 | Embedder `__new__()` 绕过正常构造，类型检查器无法推断返回类型 | `embedders/bge_m3.py:54-67` | 应改用工厂函数 + Union 注解 |
| 9 | Section filter LIKE 通配符 — `%` / `_` 在用户输入中会被解释为 SQL wildcard | `storage/sqlite_vec.py:192` | 需转义后拼接 |
| 10 | Chunker `chunk()` 的 `section_path` 参数标注"Ignored"但保留在签名中 | `chunkers/__init__.py:70` | API 误导，应移除或真正支持覆盖 |
| 11 | Embedder 无维度校验 — 错误模型返回非 1024 维会静默产生错误结果 | `embedders/bge_m3.py:96-98` | 构造后加一维断言 |

## P2 — 代码质量（中优先级）

| # | 问题 | 位置 | 建议 |
|---|------|------|------|
| 12 | TextCleaner / Chunker facade 类只覆写了部分特性 | `pipeline/core.py:114-142` | TextCleaner facade 测试缺失；应验证与 canonical impl 行为一致 |
| 13 | chunk_threshold_chars 按 4 字符/token 估算，CJK 文本严重失准 | `formatters/__init__.py:570` | CJK 应按 ~1 字符/token，或支持 per-language ratio config |
| 14 | _extract_tags_from_body() 硬编码英文停用词 + 英文国家名集合 | `formatters/__init__.py:321-434` | 中文文档标签提取完全失效；需多语言停用词表或接入 LLM 生成标签 |
| 15 | Chunker _split_by_sentence() 在 "Mr.", "U.S.A." 等缩写处错误切分 | `chunkers/__init__.py:330-364` | 应维护常见缩写列表，或使用更智能的分句器（如 nltk.sent_tokenize） |
| 16 | _split_by_headings() O(n*m) 循环查找 heading line | `chunkers/__init__.py:167` | 预构建 {line_idx: heading} dict，降为 O(n) |
| 17 | upsert_chunk 中重复 JSON serialization（dead code） | `storage/sqlite_vec.py:122,130-131` | 删除无用变量 |
| 18 | _load_sqlite_vec() 通过文件列表匹配第三方包，跨安装方式脆弱 | `storage/sqlite_vec.py:45-61` | 优先 importlib.import_module("sqlite_vec")，文件系统扫描仅作为 fallback |

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

## 架构优化建议

1. **统一异常体系** — 当前各模块返回约定不统一（None / empty string / raise）。建议定义 `myrag.exceptions` 包：`ParserNotFoundError`, `EmbeddingError`, `ChunkingError`, `FormattingError`。

2. **配置验证接入启动流程** — `_validate()` 目前是死代码，应在 `get_config()` 中调用并抛出自描述错误，避免下游静默异常。

3. **Hybrid search 性能优化** — 当前循环内重复计算 `vec_distance_cosine()`（O(n*m)）。应缓存 query embedding 的序列化结果，复用首次搜索的距离列。

4. **Chunker 重叠逻辑改进** — `_apply_overlap` 直接取前 chunk 末尾 N 字符，可能截断在单词中间；应向前扩展至最近空格边界。

5. **清理 `get_config_lazy()` 冗余** — 与 `get_config()` 功能完全相同，仅一处调用且无 circular import 问题，可直接替换为 `get_config`。

## 执行建议

```
Phase 1 (P0, ~2h): 修复死代码 + executor shutdown + FTS5 sync + debug test
Phase 2 (P1, ~4h): JSON resilience + embedder factory refactor + section filter escaping
Phase 3 (P2, ~6h): CJK token estimation + tag extraction i18n + chunker perf
Phase 4 (测试, ~4h): 补全 formatter/storage/chunker 关键路径覆盖
```

总计约 **16 小时**工作量。
