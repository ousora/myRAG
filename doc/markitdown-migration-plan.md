# MarkItDown + Trafilatura 迁移计划

## 目标

统一 parser pipeline，提升文档解析质量：
- PDF/DOCX/md/txt → MarkItDown（通用文档转换）
- HTML → Trafilatura（中文网页正文提取更准）
- 去掉各专用 parser 的重复逻辑

---

## Phase 1: 安装依赖

```bash
pip install markitdown trafilatura lxml beautifulsoup4
```

---

## Phase 2: Parser Dispatcher 改造

**文件**: `parsers/dispatcher.py`

### 改动点
- 新增 `MarkItDownParser` 类（统一处理 pdf/docx/md/txt）
- 移除各专用 parser 的 import（pdf.py/docx.py/html.py/md_parser.py）
- `resolve_parser()` 路由逻辑改为：
  - html/htm → TrafilaturaParser()
  - pdf/docx/markdown/txt → MarkItDownParser()
  - 其他扩展名 → 返回 None

### 代码结构（伪代码）

```python
# dispatcher.py — ~80 lines total

@runtime_checkable
class TextParser(Protocol):
    def parse(self, filepath: str) -> str: ...


PARSERS: dict[str, type[TextParser]] = {}


def register_parser(extension: str, parser_cls: type[TextParser]) -> None:
    PARSERS[extension] = parser_cls
```

新增两个 Parser 类：
- `TrafilaturaParser` — html/htm → extract_text() + best_match()
- `MarkItDownParser` — pdf/docx/md/txt → markitdown.MarkItDown().convert(path)

---

## Phase 3: TextCleaner（新模块）

**文件**: `parsers/text_cleaner.py`（新增，~100 lines）

### 功能
- 去空行、合并重复段落
- 通过 YAML config 加载过滤规则（正则表达式），支持用户追加自定义规则
- 去除多余空白和缩进
- 统一换行为 `\n\n`（段落级分隔）

### API
```python
class TextCleaner:
    def clean(self, text: str) -> str: ...
    
# Load rules from YAML config (default + user overrides)
cleaned = TextCleaner.clean(raw_text, rules_config="clean_rules.yaml")
```

### 默认规则（内置，可被覆盖）
- `^\s*##.*\n+` — 删除 chunk 开头的 Markdown header  
- `\s{2,}` → ` ` — 合并多余空白
- `[^\S\n]\s+[^\S\n]` — 清理行首尾空格

### YAML config 示例（clean_rules.yaml）
```yaml
rules:
  - pattern: '\[导图\]'           # Chinese anchor markers (CNAPS doc specific)
    flags: re.IGNORECASE
  - pattern: '^\s*(第一篇|第二篇)' # Chinese article numbers  
  - pattern: '(?m)^# .+\n+'      # Remove leading H1 from chunks
```

> YAML 选择原因：正则表达式在 YAML single-quote 中原样保留（无需双重转义），且支持 `#` 注释用于规则文档。JSON 需要额外一层 `\` 转义，维护成本高。

---

## Phase 5: Pipeline Integration

**文件**: `pipeline.py`

### 改动点
1. `process_file_with_md()` 中增加 TextCleaner 步骤：
```python
raw_text = parser.parse(filepath)
cleaned = TextCleaner.clean(raw_text)   # ← 新增
result = format_text_async(cleaned, source_type=source_type).result(timeout=300)
```

2. `format_text` 的 prompt 更新（prompts.py）：
- "输入已是清洗后的文本，直接结构化即可"
- 移除对广告/导航栏清理的说明（已由 TextCleaner 处理）

---

## Phase 6: Prompt + Writer 优化（之前已做部分）

**文件**: `formatters/prompts.py` — 已完成
- 约束 section_path 只包含纯章节名，不含"第一篇""导图"等标记
- CRITICAL 指令：chunk text 不包含任何锚点标记

**文件**: `formatters/writer.py` — 需要简化
- 移除 `_strip_section_header()`（LLM 不再输出脏数据）
- `_render_section_path()` 保留，过滤 generic container sections

---

## Phase 7: 测试验证

### Unit Tests
1. **TextCleaner** — mock input → verify whitespace normalization only (no content deletion)
2. **MarkItDownParser** — test file → verify markdown output
3. **TrafilaturaParser** — test HTML → extract correct body text
4. **End-to-end** — small doc → pipeline → valid .md with headers

### Regression Tests
- 重新跑 cnaps.txt（18K+ words）— 验证 JSON 不再截断
- ISO 20022 PDF — 验证中文内容正确提取
- Confluence page footer test: verify "Confidential" / "Last modified" in chunk text is preserved

---

## Checklist

- [x] Install: markitdown, trafilatura, lxml (via .venv)
- [x] Rewrite dispatcher.py (MarkItDownParser + TrafilaturaParser)
- [x] Create parsers/text_cleaner.py (~120 lines with YAML config support)
- [x] Delete old parsers: pdf.py, docx.py, html.py, md_parser.py
- [x] Update pipeline.py — TextCleaner facade delegates to new module; dispatcher auto-registers at load time
- [ ] Simplify writer.py (remove _strip_section_header)
- [ ] Run tests on cnaps.txt and ISO 20022 PDF

## Additional changes made during migration

1. **pyproject.toml** — Added `[tool.setuptools.packages.find]` to fix editable install (flat-layout discovery issue).
2. **cleaners/__init__.py** — Now delegates to `parsers.text_cleaner.TextCleaner` for backward compatibility.
3. **Test scripts updated**:
   - `.test/scripts/run_cnaps_test.py` → uses dispatcher + TextCleaner facade
   - `.test/scripts/run_pdf_test.py` → same pattern
   - `.test/scripts/run_pdf_e2e.py` → same pattern

## Risks & Mitigations

1. **Trafilatura best_match() on non-standard HTML** — fallback to extract_text() raw mode ✅ (already in TrafilaturaParser)
2. **TextCleaner regex coverage** — keep minimal; test with real English docs from Confluence/SharePoint ✅ (tests pass)
3. **Backward compatibility** — `cleaners` and pipeline TextCleaner both delegate to canonical implementation ✅
