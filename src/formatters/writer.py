"""Write formatted results to markdown files."""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Serializes the collision-check → file-write pair across threads. Without it,
# two threads processing same-titled documents could both observe the target
# path as free (TOCTOU) and overwrite each other's output.
_MD_WRITE_LOCK = threading.Lock()


def _insert_wikilinks(body: str, entities: list[dict[str, Any]]) -> str:
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
    replacements: list[tuple[int, int, str]] = []
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
    for m in re.finditer(r"```[\s\S]*?```", text):
        protected.append((m.start(), m.end()))

    # Inline code: `...` (single backtick pairs, not nested)
    for m in re.finditer(r"(?<!`)`(?!`)([^`]*)`(?!`)", text):
        protected.append((m.start(), m.end()))

    # Existing wikilinks: [[...]]
    for m in re.finditer(r"\[\[.*?\]\]", text):
        protected.append((m.start(), m.end()))

    # Existing markdown links: [text](url)
    for m in re.finditer(r"\[[^\]]*\]\([^)]*\)", text):
        protected.append((m.start(), m.end()))

    return sorted(protected)


def _is_inside_protected(position: int, protected_ranges: list[tuple[int, int]]) -> bool:
    """Check if a character position falls within any protected range."""
    return any(start <= position < end for start, end in protected_ranges)


def write_to_md(result: dict[str, Any], output_dir: str | Path) -> str:
    """Format structured result into markdown and save it.

    Two distinct documents sharing a title never overwrite each other: when
    the target file exists with *different* content, a numeric suffix
    (``-1``, ``-2``, …) is appended. Re-writing identical content keeps the
    same path, so re-running the pipeline stays idempotent.

    Args:
        result: Output from format_text() with title, tags, metadata, body
        output_dir: Directory to save the .md file (created if needed).

    Returns:
        Absolute path of the written file.

    """
    output_path = Path(output_dir).resolve()
    os.makedirs(output_path, exist_ok=True)

    if not result.get("title"):
        _msg = "Missing 'title' in formatter output"
        raise ValueError(_msg)

    title = result["title"]
    safe_name = _safe_filename(title)
    base_path = output_path / f"{safe_name}.md"

    metadata = result.get("metadata", {})

    lines: list[str] = []

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
        stripped_body = re.sub(r"^#\s+.*\n", "", body, count=1).strip()

        # Apply wikilinks for .md display only (entities extracted by formatter)
        entities = metadata.get("entities", [])
        if entities:
            stripped_body = _insert_wikilinks(stripped_body, entities)

        _write_body_with_sections(lines, stripped_body, sections)

    md_content = "\n".join(lines).rstrip() + "\n"

    with _MD_WRITE_LOCK:
        file_path = _resolve_output_path(base_path, md_content)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
        except (OSError, PermissionError) as exc:  # noqa: BLE001 — log and re-raise with context
            _err_msg = f"Failed to write markdown file to {file_path}: {exc}"
            raise OSError(_err_msg) from exc

    return file_path


def _resolve_output_path(base_path: Path, md_content: str) -> str:
    """Pick the write target for *base_path*, avoiding clobbering other docs.

    - Path free → use it.
    - Path exists with byte-identical content → reuse it (idempotent re-runs).
    - Path exists with different content → append ``-N`` before ``.md``.
    """
    if not base_path.exists() or base_path.read_text(encoding="utf-8") == md_content:
        return str(base_path)

    logger.warning("Markdown collision: %s exists with different content", base_path.name)
    stem, suffix = base_path.stem, base_path.suffix
    for n in range(1, 1000):
        candidate = base_path.with_name(f"{stem}-{n}{suffix}")
        if not candidate.exists() or candidate.read_text(encoding="utf-8") == md_content:
            return str(candidate)
    _msg = f"Could not find a free filename for {base_path.name} (tried 999 suffixes)"
    raise OSError(_msg)


def _safe_filename(title: str) -> str:
    """Generate a safe filename from title.

    Preserves Unicode characters (UTF-8 paths are standard on modern systems).
    Only removes characters that are truly problematic in filenames.

    Args:
        title: Document title string

    Returns:
        Safe filename without extension, never empty.

    """
    # Remove only characters that cause issues across all filesystems
    safe = re.sub(r'[/\\:*?"<>|]', "_", title)
    safe = safe.strip() or "untitled"
    if len(safe) > 200:
        safe = safe[:200]
    return safe


def _write_yaml_frontmatter(lines: list[str], result: dict[str, Any]) -> None:
    """Write a YAML front matter block.

    Fields written (when present and non-empty):
      - title     — document title
      - source_file — original document path
      - created_at  — ISO-8601 timestamp of ingestion
      - modified_date — last modification date (if available)
      - tags        — list of tag strings

    Values are serialized with ``yaml.safe_dump`` so quoting/escaping follows
    the YAML spec instead of Python ``repr()`` heuristics.
    """
    metadata = result.get("metadata", {})
    fields: list[tuple[str, object]] = []

    if title := result.get("title"):
        fields.append(("title", title))
    if source_file := (metadata.get("source_file") or ""):
        fields.append(("source", source_file))
    if created_at := (metadata.get("created_at") or ""):
        fields.append(("created_at", created_at))
    if modified_date := metadata.get("modified_date"):
        fields.append(("modified_date", modified_date))
    tags = result.get("tags", [])
    if tags:
        fields.append(("tags", [str(t) for t in tags]))

    for key, value in fields:
        dumped = yaml.safe_dump({key: value}, allow_unicode=True, default_flow_style=False)
        # safe_dump of a single-key mapping emits "key: value\n" (or a block
        # list for tags); append it verbatim, stripping the trailing newline.
        lines.append(dumped.rstrip("\n"))


def _write_metadata_block(lines: list[str], result: dict[str, Any]) -> None:
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


def _write_body_with_sections(lines: list[str], body: str, sections: list[dict[str, Any]]) -> None:
    """Write body content to the output.

    The LLM formatter produces valid markdown (headings, tables, code blocks).
    We write it as-is without splitting — any header manipulation is handled
    by ``_render_markdown_with_sections`` in pipeline.core when needed.
    """
    lines.append("")
    lines.append(body.strip())


def format_md(result: dict[str, Any]) -> str:
    """Format result into markdown string (no file write)."""
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="myrag_md_"))
    path = write_to_md(result, str(tmp_dir))
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    finally:
        # Clean up the temp directory and its file.
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


__all__ = ["format_md", "write_to_md"]
