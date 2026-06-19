"""Write formatted results to markdown files."""

import os
import re


def _insert_wikilinks(body: str, entities: list[dict]) -> str:
    """Replace entity mentions with [[wikiname]] format for .md display.

    Only called from write_to_md() — never used in the chunk/embed pipeline.
    The chunker and embedder always receive clean text without wikilinks.

    CRITICAL: Skips code blocks, inline code, and existing links to avoid corruption.
    Uses longest-match-first to prevent short entity names from overwriting long ones.
    Applies replacements back-to-front to prevent string offset shift bugs.
    """
    if not entities:
        return body

    # 1. Extract protected ranges (code blocks, inline code, existing links)
    protected_ranges = _extract_protected_ranges(body)

    # 2. Collect all replacements from longest to shortest entity name
    replacements = []
    for e in sorted(entities, key=lambda x: -len(x["name"])):
        pattern = re.escape(e["name"])
        for match in re.finditer(pattern, body):
            pos_start, pos_end = match.start(), match.end()
            if not _is_inside_protected(pos_start, protected_ranges):
                replacement = f'[[{e["name"]}]]'
                # Skip if this position was already claimed by a longer entity
                if not any(ps <= pos_start < pe for ps, pe, _ in replacements):
                    replacements.append((pos_start, pos_end, replacement))

    # 3. Apply back-to-front so earlier positions stay valid
    for start, end, replacement in sorted(replacements, key=lambda x: -x[0]):
        body = body[:start] + replacement + body[end:]

    return body


def _extract_protected_ranges(text: str) -> list[tuple[int, int]]:
    """Find all protected regions where wikilink insertion is unsafe.

    Returns sorted list of (start, end) tuples covering:
    - Code blocks (```...``` with optional language tag)
    - Inline code (`...`)
    - Existing wikilinks ([[...]])
    - Existing markdown links ([text](url))
    """
    protected = []

    # Fenced code blocks: ``` ... ``` (including language tag)
    for m in re.finditer(r'```[\s\S]*?```', text):
        protected.append((m.start(), m.end()))

    # Inline code: `...` (single backtick pairs, not nested)
    for m in re.finditer(r'(?<!`)`(?!`)([^`]*)`(?!`)', text):
        protected.append((m.start(), m.end()))

    # Existing wikilinks: [[...]]
    for m in re.finditer(r'\[\[.*?\]\]', text):
        protected.append((m.start(), m.end()))

    # Existing markdown links: [text](url)
    for m in re.finditer(r'\[[^\]]*\]\([^)]*\)', text):
        protected.append((m.start(), m.end()))

    return sorted(protected)


def _is_inside_protected(position: int, protected_ranges: list[tuple[int, int]]) -> bool:
    """Check if a character position falls within any protected range."""
    for start, end in protected_ranges:
        if start <= position < end:
            return True
    return False


def write_to_md(result, output_dir):
    """Format structured result into markdown and save it.

    Args:
        result: Output from format_text() with title, tags, metadata, body
        output_dir: Directory to save the .md file (created if needed).

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Validate required fields before writing
    if not result.get("title"):
        raise ValueError(f"Missing 'title' in formatter output")

    title = result["title"]
    safe_name = _safe_filename(title)
    file_path = os.path.join(output_dir, f"{safe_name}.md")

    metadata = result.get("metadata", {})

    lines = []

    # YAML front matter (standard format for Obsidian, VS Code)
    _write_yaml_frontmatter(lines, result)
    lines.append("---")
    lines.append("")

    # Title with blank line after
    lines.append(f"# {title}")
    lines.append("")

    # Structured metadata block (word count, section outline)
    _write_metadata_block(lines, result)

    # Body content — strip the first H1 since we already have a title above
    body = result.get("body", "")
    if body and isinstance(body, str) and body.strip():
        sections = metadata.get("sections", [])
        # Remove the first H1 heading (e.g., "# China National Clearing Center")
        # since we already render it above as the document title.
        stripped_body = re.sub(r'^#\s+.*\n', '', body, count=1).strip()

        # Apply wikilinks for .md display only (entities extracted by formatter)
        entities = metadata.get("entities", [])
        if entities:
            stripped_body = _insert_wikilinks(stripped_body, entities)

        _write_body_with_sections(lines, stripped_body, sections)

    md_content = "\n".join(lines).rstrip() + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return file_path


def _safe_filename(title):
    """Generate a safe filename from title.

    Preserves Unicode characters (UTF-8 paths are standard on modern systems).
    Only removes characters that are truly problematic in filenames.

    Args:
        title: Document title string

    Returns:
        Safe filename without extension
    """
    # Remove only characters that cause issues across all filesystems
    safe = re.sub(r'[/\\:*?"<>|]', '_', title)
    return safe.strip()


def _write_yaml_frontmatter(lines, result):
    """Write YAML front matter block.

    Fields written (when present and non-empty):
      - title     — document title
      - source_file — original document path
      - created_at  — ISO-8601 timestamp of ingestion
      - modified_date — last modification date (if available)
      - tags        — list of tag strings
    """
    metadata = result.get("metadata", {})
    source_file = metadata.get("source_file") or ""
    created_at = metadata.get("created_at") or ""
    modified_date = metadata.get("modified_date")
    tags = result.get("tags", [])

    lines.append("---")
    if title := result.get("title"):
        lines.append(f"title: {repr(title)}")  # use repr for safe YAML string quoting
    if source_file:
        lines.append(f"source: {repr(source_file)}")
    if created_at:
        lines.append(f"created_at: {created_at}")
    if modified_date:
        lines.append(f"modified_date: {repr(modified_date)}")
    if tags:
        # YAML list format for tags
        lines.append("tags:")
        for tag in tags:
            lines.append(f'  - "{tag}"')


def _write_metadata_block(lines, result):
    """Write a structured metadata block.

    Fields written (when present and non-empty):
      - Words — word count (tags are already in YAML front matter)
      - Sections — numbered heading outline with levels
    """
    metadata = result.get("metadata", {})
    total_words = metadata.get("total_words")
    sections_list = metadata.get("sections", [])

    meta_lines: list[str] = []

    # Word count
    if total_words:
        meta_lines.append(f"- **Words:** {total_words}")

    # Section outline (numbered, starting from 1)
    if sections_list:
        section_items = []
        for idx, s in enumerate(sections_list, start=1):
            level = s.get("level", 2)
            indent = "  " * max(level - 3, 0)  # H2/H3 get no extra indent beyond base
            section_items.append(f"{idx}. {indent}**{s['title']}**")
        meta_lines.append("")
        meta_lines.append("- **Sections:**")
        for item in section_items:
            meta_lines.append(item)

    # Blank line before and after metadata block (markdown paragraph separation)
    if meta_lines:
        lines.append("")
        lines.extend(meta_lines)
        lines.append("")


def _write_body_with_sections(lines, body: str, sections: list):
    """Write body content to the output.

    The LLM formatter produces valid markdown (headings, tables, code blocks).
    We write it as-is without splitting — any header manipulation is handled
    by ``_render_markdown_with_sections`` in pipeline.core when needed.
    """
    lines.append("")
    lines.append(body.strip())


def format_md(result):
    """Format result into markdown string (no file write)."""
    os.makedirs("/tmp/md_format_output", exist_ok=True)
    path = write_to_md(result, "/tmp/md_format_output")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


__all__ = ["format_md", "write_to_md"]
