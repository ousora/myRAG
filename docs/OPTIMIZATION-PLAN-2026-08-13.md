# myRAG 优化计划

> 基于完整代码审查、配置审计和依赖分析，按优先级排序的优化项。每项包含：文件位置、问题描述、影响、修复方案、预估工作量。

---

## 高优先级（影响功能正确性或稳定性）

### H1: 修复 `ingest.py` 资源泄漏

- **文件**: `src/pipeline/ingest.py:56-79`
- **问题**: `Embedder()` 创建的 `httpx.Client` 和 `SQLiteVecStore()` 创建的数据库连接从不关闭，每次调用都泄漏网络客户端和数据库连接。
- **影响**: 长时间运行或批量处理时，文件描述符和内存持续增长，最终导致 `Too many open files` 错误。
- **修复方案**:
  ```python
  # 使用 with 语句或 try/finally
  with Embedder() as e:
      stored_chunks = e.store_chunks(all_chunks, doc_id=doc_id)
      stored_doc = e.store_document(...)
  
  with SQLiteVecStore(store_path) as db:
      db.upsert_chunks(stored_chunks, doc_id=doc_id)
      db.upsert_document(...)
  ```
  同时需为 `Embedder` 和 `SQLiteVecStore` 添加 `__enter__`/`__exit__` 支持（`Embedder` 已有 `__exit__` 但缺少 `__enter__`）。
- **预估工作量**: 30 分钟

### H2: 修复 `cli.py` 缺失 `hybrid` 子命令注册

- **文件**: `src/pipeline/cli.py:18-50`
- **问题**: `elif args.command == "hybrid"` 分支（第99行）存在，但 `subparsers` 中没有调用 `subparsers.add_parser("hybrid", ...)`。运行 `myrag hybrid` 会静默无操作。
- **影响**: 用户无法通过 CLI 使用 hybrid 模式。
- **修复方案**: 在第49行（`md_parser` 之后）添加：
  ```python
  hybrid_parser = subparsers.add_parser("hybrid", help="Process file with hybrid indexing")
  hybrid_parser.add_argument("input", help="file to process")
  hybrid_parser.add_argument("--store", required=True, help="Path to sqlite-vec database")
  ```
- **预估工作量**: 10 分钟

### H3: 修复 `search_documents` 未使用 `query_vector` 参数

- **文件**: `src/storage/search.py:122-180`
- **问题**: `search_documents()` 接受 `query_vector` 参数，但当提供向量时没有将其用于排序。第149-158行的 SQL 中，`query_vector is not None` 条件下虽然构建了 `emb_str`，但 `WHERE` 子句只添加了 `embedding IS NOT NULL`，没有 `ORDER BY vec_distance_cosine`。
- **影响**: `rag_query` 中调用 `store.search_documents(query_vector=query_vector, k=1)` 获取文档级上下文时，返回结果不按相似度排序，而是按数据库插入顺序。
- **修复方案**:
  ```python
  if query_vector is not None:
      where_clauses.append("embedding IS NOT NULL")
      # 在 ORDER BY 中使用向量距离
      sql = f"""SELECT ..., vec_distance_cosine(embedding, ?) AS distance
                FROM documents {where}
                ORDER BY distance ASC
                LIMIT ?"""
      rows = self.conn.execute(sql, [emb_str, *params, k]).fetchall()
  ```
- **预估工作量**: 20 分钟

### H4: 修复 `formatters/__init__.py` 线程池泄漏

- **文件**: `src/formatters/__init__.py:130-138, 33-46`
- **问题**: 全局 `_executor` 通过 `get_executor()` 懒初始化，但 `atexit` 回调中 `_shutdown_executor()` 存在逻辑错误——`_executor` 是模块级变量，`_shutdown_executor` 中引用的是全局变量但 `global _executor` 声明正确。然而，如果 `get_executor()` 从未被调用，`_executor` 为 `None`，`atexit` 正常返回；如果被调用过，`shutdown(wait=True)` 会等待所有未完成的任务，但如果主线程被 `Ctrl+C` 中断，`atexit` 可能来不及执行。
- **影响**: 后台线程持续占用内存和 CPU（即使没有任务），进程退出时可能不优雅。
- **修复方案**: 保持现有 `atexit` 机制，但添加 `shutdown()` 公共方法供外部调用：
  ```python
  def shutdown_executor():
      """Explicitly shut down the formatter thread pool."""
      global _executor
      if _executor is not None:
          _executor.shutdown(wait=False)  # 非阻塞，避免卡住
          _executor = None
  ```
- **预估工作量**: 15 分钟

### H5: 为 `Embedder` 和 `SQLiteVecStore` 添加 `__enter__` 支持

- **文件**: `src/embedders/bge_m3.py:144-153`, `src/storage/sqlite_vec.py:64-74`
- **问题**: `Embedder.__exit__` 已实现但缺少 `__enter__`；`SQLiteVecStore` 只有 `close()` 没有上下文管理器协议。两者都无法使用 `with` 语句。
- **影响**: 无法用 `with` 安全地管理资源生命周期。
- **修复方案**:
  ```python
  # bge_m3.py — 添加 __enter__
  def __enter__(self):
      return self
  
  # sqlite_vec.py — 添加 __enter__ 和 __exit__
  def __enter__(self):
      return self
  
  def __exit__(self, *exc_info):
      self.close()
      return False
  ```
- **预估工作量**: 15 分钟

---

## 中优先级（影响性能或代码质量）

### M1: 提取 `call_llm` 和 `call_llm_raw` 的共享请求逻辑

- **文件**: `src/formatters/__init__.py:141-285`
- **问题**: `call_llm` 和 `call_llm_raw` 重复构造 payload、调用 `httpx.post`、处理 `httpx.HTTPError`、解析 `response.json()["choices"][0]["message"]["content"]`。DRY 违规，且 `call_llm_raw` 没有 schema fallback 逻辑。
- **影响**: 修改请求逻辑（如添加 header、修改超时）需要同步修改两处。
- **修复方案**: 提取 `_call_llm_api(payload, timeout)` 私有函数：
  ```python
  def _call_llm_api(payload: dict, timeout: int | None) -> httpx.Response:
      """Make HTTP POST to LLM endpoint with error handling."""
      cfg = _get_config()
      try:
          response = httpx.post(cfg.llm_endpoint, json=payload, timeout=timeout or cfg.llm_timeout)
          response.raise_for_status()
          return response
      except httpx.HTTPError as e:
          logger.error("LLM call failed after %.1fs: %s", (timeout or cfg.llm_timeout), e)
          raise RuntimeError(f"LLM API request failed: {e}") from e
  ```
  然后 `call_llm` 和 `call_llm_raw` 都调用此函数。
- **预估工作量**: 30 分钟

### M2: 预编译 `re.split(r'\n\n+', text)` 正则

- **文件**: `src/formatters/__init__.py:429`
- **问题**: `_split_by_paragraph` 每次调用 `format_text` 时都编译 `r'\n\n+'` 正则。
- **影响**: 小开销，但在高频调用场景下可累积。
- **修复方案**:
  ```python
  _PARAGRAPH_SPLIT = re.compile(r'\n\n+')
  # 第429行改为：
  paragraphs = _PARAGRAPH_SPLIT.split(text)
  ```
- **预估工作量**: 5 分钟

### M3: 修复 `upsert_document` 的 SELECT+INSERT/UPDATE 非原子操作

- **文件**: `src/storage/inserts.py:201-217`
- **问题**: 先用 `SELECT` 检查是否存在，再 `INSERT` 或 `UPDATE`。并发场景下可能产生重复记录。
- **影响**: 多进程/多线程写入同一数据库时可能产生重复文档记录。
- **修复方案**: 使用 SQLite 的 `INSERT OR REPLACE` 或 `INSERT ... ON CONFLICT`：
  ```sql
  INSERT INTO documents (title, tags, text_summary, source_file, total_chunks, embedding, created_at)
  VALUES (?, ?, ?, ?, ?, ?, ?)
  ON CONFLICT(source_file) DO UPDATE SET
    title=excluded.title, tags=excluded.tags, text_summary=excluded.text_summary,
    total_chunks=excluded.total_chunks, embedding=excluded.embedding, created_at=excluded.created_at
  ```
  需在 `documents` 表上添加 `UNIQUE(source_file)` 约束（当前无）。
- **预估工作量**: 30 分钟

### M4: 修复 `search.py` 中 `_build_fts_query` 重复编译 CJK 正则

- **文件**: `src/storage/search.py:45-47`
- **问题**: `"|".join(_CJK_RANGE)` 被编译两次——第45行用于 `re.findall`，第47行又编译一次 `re.compile`。
- **影响**: 每次 FTS 查询浪费一次正则编译。
- **修复方案**:
  ```python
  _CJK_PAT = re.compile("|".join(_CJK_RANGE))
  # 第45-47行改为：
  tokens = re.findall(r"[A-Za-z0-9]+|" + "|".join(_CJK_RANGE), cleaned)
  tokens = [t for t in tokens if len(t) > 1 or bool(_CJK_PAT.match(t))]
  ```
- **预估工作量**: 5 分钟

### M5: 修复 `inserts.py` 中 `_count_words` 逐字符迭代性能

- **文件**: `src/storage/inserts.py:19-38`
- **问题**: 每个字符单独判断是否 CJK，然后重建非 CJK 字符串再计数。对于每个 chunk 都执行一次，100 个 chunk 就是 100 次全量迭代。
- **影响**: 大文档批量处理时，词数统计成为瓶颈。
- **修复方案**:
  ```python
  _CJK_RE = re.compile("|".join(_CJK_RANGE))  # 已在第16行预编译
  def _count_words(text: str) -> int:
      if not text:
          return 0
      non_cjk = re.sub(_CJK_RE, '', text)
      return len(re.findall(r'\S+', non_cjk)) + len(_CJK_RE.findall(text))
  ```
- **预估工作量**: 15 分钟

### M6: 修复 `config.yaml` 与 `config.example.yaml` 不一致

- **文件**: `conf/config.yaml` vs `conf/config.example.yaml`
- **问题**: `config.yaml` 缺少 `embedding.mode`、`embedding.query_instruction`、`logging.debug_log_llm_responses` 三个字段。虽然 `Config` 类有默认值，但显式缺失导致配置意图不清晰。
- **影响**: 新用户复制 `config.example.yaml` 后行为与现有 `config.yaml` 不同；`debug_log_llm_responses` 缺失意味着调试时无法保存 LLM 原始响应。
- **修复方案**: 在 `conf/config.yaml` 中补充缺失字段：
  ```yaml
  embedding:
    mode: "remote"
    base_url: "http://localhost:11435"
    model: "bge-m3"
    timeout: 60
    query_instruction: "Represent this sentence for searching relevant passages: "
  
  logging:
    max_bytes: 5242880
    debug_log_llm_responses: false
  ```
- **预估工作量**: 5 分钟

### M7: 修复 `pyproject.toml` 重复的 `[tool.pytest.ini_options]` 段

- **文件**: `pyproject.toml:31-32, 67-71`
- **问题**: 两个 `[tool.pytest.ini_options]` 段，第一段只定义了 `pythonpath`，第二段覆盖了 `testpaths`、`addopts` 等。第二段生效，但重复段令人困惑。
- **影响**: 维护者可能误在第一个段中添加配置。
- **修复方案**: 合并为单个段：
  ```toml
  [tool.pytest.ini_options]
  pythonpath = ["src"]
  testpaths = ["src"]
  python_files = ["test_*.py"]
  python_classes = ["Test*"]
  python_functions = ["test_*"]
  addopts = "-v --tb=short"
  ```
- **预估工作量**: 5 分钟

### M8: 修复 `pyproject.toml` 中 `sentence-transformers` 上限过旧

- **文件**: `pyproject.toml:24`
- **问题**: `local-embeddings = ["sentence-transformers>=2.7,<3"]`，上限 `<3` 排除了已稳定的 3.x 版本。
- **影响**: 无法使用 `sentence-transformers` 3.x 的新功能和性能改进，未来 2.x 停止维护时将阻塞安装。
- **修复方案**: 改为 `>=2.7`（移除上限）或 `>=3.0`（如果确认代码兼容 3.x）。建议先升级到 3.x 测试后再更新。
- **预估工作量**: 15 分钟（含测试验证）

### M9: 修复 `pyproject.toml` 中 `types-PyYAML` 缺失

- **文件**: `pyproject.toml:19-22`
- **问题**: `.pre-commit-config.yaml` 中 `mypy` 钩子依赖 `types-PyYAML`，但 `pyproject.toml` 的 `[project.optional-dependencies]` 的 `dev` 中未列出。本地运行 `uv run mypy` 会因缺少 PyYAML 类型存根而失败。
- **修复方案**:
  ```toml
  dev = [
      "pytest>=7.4",
      "ruff>=0.3",
      "mypy>=1.0",
      "types-PyYAML",
  ]
  ```
- **预估工作量**: 5 分钟

### M10: 修复 `cli.py` 中 `logs/` 目录创建不检查是否为文件

- **文件**: `src/pipeline/cli.py:53-54`
- **问题**: `log_dir.mkdir(exist_ok=True)` 如果 `logs` 已作为文件存在会抛出 `NotADirectoryError`。
- **影响**: 罕见但可能——用户先创建 `logs` 文件再运行 CLI。
- **修复方案**:
  ```python
  if log_dir.is_file():
      raise OSError(f"'{log_dir}' exists as a file, not a directory")
  log_dir.mkdir(exist_ok=True)
  ```
- **预估工作量**: 5 分钟

### M11: 修复 `markdown_utils.py` 标题匹配大小写敏感

- **文件**: `src/pipeline/markdown_utils.py:46`
- **问题**: `re.match(rf'^#\s+{re.escape(title)}$', stripped)` 是大小写敏感的。如果 LLM 输出的标题大小写与 `result["title"]` 不同，重复标题不会被移除。
- **影响**: 生成的 markdown 可能出现重复的标题标题行。
- **修复方案**: 添加 `re.IGNORECASE`：
  ```python
  if re.match(rf'^#\s+{re.escape(title)}$', stripped, re.IGNORECASE):
  ```
- **预估工作量**: 5 分钟

### M12: 为 `rag_query` 添加返回类型标注

- **文件**: `src/pipeline/core.py:418-420`
- **问题**: `def rag_query(...) -> dict:` 缺少返回类型。项目规范要求所有公共函数都有返回类型标注。
- **修复方案**:
  ```python
  def rag_query(question: str, db_path: str, *, k: int = 5,
                db: "SQLiteVecStore | None" = None,
                embedder: "Embedder | None" = None) -> dict:
  ```
  （已有 `-> dict`，确认即可）
- **预估工作量**: 5 分钟

### M13: 修复 `local_bge.py` 类型不一致

- **文件**: `src/embedders/local_bge.py:52-107, 124-130`
- **问题**: `LocalEmbedder.embed()` 对 str 输入返回 `list[float]`，对 list[str] 返回 `list[list[float]]`。但 `embed_query()` 的类型标注为 `-> list[list[float]]`，当传入 str 时实际返回 `list[float]`，与标注不符。
- **影响**: 调用方如果依赖 `embed_query` 返回 `list[list[float]]`，对单字符串输入会得到意外的扁平结构。
- **修复方案**: 统一 `embed_query` 的返回行为——对 str 输入包装为 `[[emb]]`，或修正类型标注为 `-> list[float] | list[list[float]]`。
- **预估工作量**: 15 分钟

### M14: 修复 `prompts.py` 中 `try_fix_common_issues` 浅拷贝

- **文件**: `src/formatters/prompts.py:338`
- **问题**: `fixed = dict(result)` 是浅拷贝。如果 LLM 返回的 `result["metadata"]` 被后续代码修改，原始 `result` 也会被影响。
- **影响**: 罕见的竞态条件，在缓存命中后修改结果时可能发生数据污染。
- **修复方案**:
  ```python
  import copy
  fixed = copy.deepcopy(result)
  ```
  或仅深拷贝嵌套结构：
  ```python
  fixed = dict(result)
  if isinstance(fixed.get("metadata"), dict):
      fixed["metadata"] = dict(fixed["metadata"])
  ```
- **预估工作量**: 10 分钟

### M15: CI 添加 mypy 步骤

- **文件**: `.github/workflows/ci.yml`
- **问题**: CI 只运行 `ruff check` 和 `pytest`，没有 `mypy`。项目配置了严格的类型检查（`disallow_untyped_defs = true`），但 PR 不会因类型错误被阻断。
- **影响**: 类型错误可能合并到 main 分支。
- **修复方案**: 在 CI 中添加 mypy 步骤：
  ```yaml
  - name: Run mypy
    run: uv run mypy src/
  ```
- **预估工作量**: 10 分钟

---

## 低优先级（改进型）

### L1: 修复 `pyproject.toml` 中 `[project.scripts]` 缺失

- **文件**: `pyproject.toml`
- **问题**: 没有 `[project.scripts]` 入口点，`pip install myrag-pipeline` 不会提供 `myrag` 命令行工具。
- **修复方案**:
  ```toml
  [project.scripts]
  myrag = "pipeline.cli:main"
  ```
- **预估工作量**: 5 分钟

### L2: 修复 `.gitignore` 缺少 `*.db`、`*.sqlite3` 模式

- **文件**: `.gitignore`
- **问题**: 只忽略了 `data/` 目录，但如果用户把 `.db` 文件放在其他目录（如 `output/`），可能被提交到 git。
- **修复方案**: 添加：
  ```
  *.db
  *.sqlite3
  *.sqlite
  ```
- **预估工作量**: 5 分钟

### L3: 修复 `formatters/cache.py` 缓存键包含完整系统提示

- **文件**: `src/formatters/cache.py:26-28`
- **问题**: 缓存键 `hashlib.md5(f"{source_type}|{sp}|{raw}".encode())` 中 `sp`（系统提示）约 5KB，导致哈希计算被系统提示主导。每次系统提示变化都会创建全新命名空间。
- **影响**: 缓存命中率偏低，因为系统提示的微小变化（如 chunked mode 的 `{chunk_label}`）就导致缓存失效。
- **修复方案**: 分别哈希各组件：
  ```python
  sp_hash = hashlib.sha256((system_prompt or "").encode()).hexdigest()
  raw_hash = hashlib.sha256(raw.encode()).hexdigest()
  return hashlib.md5(f"{source_type}|{sp_hash}|{raw_hash}".encode()).hexdigest()
  ```
- **预估工作量**: 15 分钟

### L4: 修复 `writer.py` 中 YAML frontmatter 使用 `repr()` 产生单引号

- **文件**: `src/formatters/writer.py:181-192`
- **问题**: `repr(title)` 产生 Python 风格的单引号字符串（如 `'Title'`），而非 YAML 标准的双引号（`"Title"`）。大多数 YAML 解析器接受单引号，但严格解析器可能拒绝。
- **影响**: 与 Obsidian 等工具集成时可能出现兼容性警告。
- **修复方案**: 使用双引号或 `yaml.safe_dump`：
  ```python
  import yaml
  lines.append(f"title: {yaml.safe_dump(title, default_style='\"').strip()}")
  ```
- **预估工作量**: 15 分钟

### L5: 修复 `embedders/bge_m3.py` 中 `httpx.TimeoutException` 在 httpx 1.0+ 已移除

- **文件**: `src/embedders/bge_m3.py:177`
- **问题**: `httpx.TimeoutException` 在 httpx 0.28+ 标记为弃用，在 1.0+ 已移除。正确异常是 `httpx.ReadTimeout`/`httpx.WriteTimeout` 或 `httpx.TimeoutException`（仍在但指向子异常）。
- **影响**: 升级到 httpx 1.0+ 后嵌入重试逻辑会崩溃。
- **修复方案**:
  ```python
  except (httpx.TimeoutException, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
  ```
  或更通用：
  ```python
  except httpx.TransportError as exc:
  ```
- **预估工作量**: 10 分钟

### L6: 修复 `formatters/__init__.py` 中 `import hashlib` 不在模块顶部

- **文件**: `src/formatters/__init__.py:214-217`
- **问题**: `import hashlib` 在 `call_llm` 函数体内（仅在 `debug_log_llm_responses` 为 True 时执行），不符合项目"顶部导入"的约定。
- **影响**: 首次启用调试日志时才会导入 hashlib，行为不一致。
- **修复方案**: 将 `import hashlib` 移到文件顶部导入区（第7-15行）。
- **预估工作量**: 5 分钟

### L7: 修复 `chunkers/__init__.py` 中 `_parse_headings` 每次调用创建新的 `MarkdownIt` 实例

- **文件**: `src/chunkers/__init__.py:123-124`
- **问题**: `md = MarkdownIt()` 在 `_parse_headings` 中，每次 `chunk()` 调用都创建新实例。`MarkdownIt` 初始化较昂贵。
- **影响**: 批量处理多个文档时，每个文档的每个 chunk 都重新创建解析器。
- **修复方案**: 在 `__init__` 中创建一次并复用：
  ```python
  def __init__(self, ...):
      ...
      self._md = MarkdownIt()  # 创建一次
  
  def _parse_headings(self, text):
      tokens = self._md.parse(text)  # 复用实例
  ```
- **预估工作量**: 10 分钟

### L8: 修复 `embedders/bge_m3.py` 中 `store_chunks` 将所有文本一次性发送到嵌入 API

- **文件**: `src/embedders/bge_m3.py:238-253`
- **问题**: `self.embed(texts)` 将所有 chunk 文本一次性发送。对于 100+ chunks 的文档，每个 chunk 几 KB，总载荷可能数百 KB，可能超过 API 的最大请求大小。
- **影响**: 大文档嵌入失败，整个批次回退到无嵌入模式。
- **修复方案**: 添加 `batch_size` 参数并分批发送：
  ```python
  def store_chunks(self, chunks, *, doc_id="doc_0", batch_size=32):
      results = []
      for i in range(0, len(chunks), batch_size):
          batch = chunks[i:i+batch_size]
          texts = [c["text"] for c in batch]
          embeddings = self.embed(texts) if texts else []
          for j, chunk in enumerate(batch):
              ...
  ```
- **预估工作量**: 20 分钟

### L9: 修复 `storage/search.py` 中 `hybrid_search` 的 `IN` 子句无上限

- **文件**: `src/storage/search.py:254-258, 282-287, 308-311`
- **问题**: `",".join("?" * len(ids_to_score))` 为每个 ID 创建一个占位符。如果向量搜索结果返回 10000+ 条，会超过 SQLite 的 `SQLITE_MAX_VARIABLE_NUMBER` 限制（默认 32766）。
- **影响**: 大索引上搜索时，`hybrid_search` 可能因 SQL 变量数超限而失败。
- **修复方案**: 限制候选数量：
  ```python
  MAX_IN_CLAUSE = 1000  # 留有余量
  ids_to_score = ids_to_score[:MAX_IN_CLAUSE]
  ```
- **预估工作量**: 10 分钟

### L10: 修复 `pipeline/core.py` 中 `process_file` 和 `process_file_hybrid` 返回类型不一致

- **文件**: `src/pipeline/core.py:112-127, 130-150`
- **问题**: `process_file` 在无 parser 时返回 `[]`（空列表），`process_file_hybrid` 返回 `{"chunks": [], "document": {}, ...}`（空结构）。调用方需要检查返回类型。
- **影响**: 调用方必须知道调用的是哪个函数才能正确处理返回值。
- **修复方案**: 统一返回类型——`process_file` 也返回结构化的结果字典，或在文档中明确说明差异。
- **预估工作量**: 20 分钟

---

## 修复顺序建议

按以下顺序执行，每项完成后运行 `uv run pytest -v` 和 `uv run ruff check .` 验证：

| 顺序 | 修复项 | 原因 |
|------|--------|------|
| 1 | H2 (cli hybrid) | 最小改动，修复可见功能缺失 |
| 2 | M6 (config 一致性) | 最小改动，改善配置清晰度 |
| 3 | M7 (pytest 重复段) | 最小改动，清理配置 |
| 4 | M9 (types-PyYAML) | 使本地 mypy 可用 |
| 5 | H1 (ingest 资源泄漏) | 修复关键稳定性问题 |
| 6 | H3 (search_documents 向量) | 修复 RAG 查询正确性 |
| 7 | H5 (上下文管理器) | 支持 H1 的 with 用法 |
| 8 | M4 (CJK 正则重复编译) | 小优化 |
| 9 | M5 (_count_words 性能) | 小优化 |
| 10 | M11 (标题大小写) | 小修复 |
| 11 | M12 (rag_query 返回类型) | 类型检查 |
| 13 | M13 (local_bge 类型) | 类型检查 |
| 14 | M14 (浅拷贝) | 数据一致性 |
| 15 | M15 (CI mypy) | 质量保障 |
| 16 | M8 (sentence-transformers) | 依赖更新（需测试验证） |
| 17 | M1 (DRY call_llm) | 代码质量 |
| 18 | M2 (正则预编译) | 小优化 |
| 19 | M3 (upsert 原子性) | 并发安全 |
| 20 | M10 (logs 文件检查) | 健壮性 |
| 21 | L1 (scripts 入口) | 可用性 |
| 22 | L2 (.gitignore) | 清理 |
| 23 | L3 (缓存键) | 性能 |
| 24 | L4 (YAML 引号) | 兼容性 |
| 25 | L5 (httpx 兼容性) | 未来兼容 |
| 26 | L6 (import 位置) | 代码风格 |
| 27 | L7 (MarkdownIt 复用) | 性能 |
| 28 | L8 (store_chunks 分批) | 大文档支持 |
| 29 | L9 (IN 子句上限) | 大索引支持 |
| 30 | L10 (返回类型统一) | API 一致性 |
