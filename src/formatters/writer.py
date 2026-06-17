"""Write formatted results to markdown files."""

import hashlib
import os
import re


def write_to_md(result, output_dir):
    """Format structured result into markdown and save it.

    Args:
        result: Output from format_text() with title, tags, metadata, body
        output_dir: Directory to save the .md file (created if needed).

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    title = result["title"]
    safe_name = _safe_filename(title)
    file_path = os.path.join(output_dir, f"{safe_name}.md")

    metadata = result.get("metadata", {})

    lines = []
    # Title with blank line after (required for proper markdown rendering)
    lines.append(f"# {title}")
    lines.append("")

    # Structured YAML-style metadata block
    _write_metadata_block(lines, result)

    # Body content
    body = result.get("body", "")
    if body and isinstance(body, str) and body.strip():
        sections = metadata.get("sections", [])
        _write_body_with_sections(lines, body, sections)

    md_content = "\n".join(lines).rstrip() + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return file_path


def _safe_filename(title):
    """Generate a safe filename from title.

    Keeps ASCII alphanumerics and basic punctuation; replaces others with underscore.
    Appends short hash suffix for titles containing non-ASCII characters
    (e.g., Chinese, Japanese) to avoid collisions and unreadable filenames.
    """
    # Keep ASCII alphanumerics plus hyphen, dot, space (for readability)
    safe = "".join(c if c.isascii() and (c.isalnum() or c in "-_. ") else "_" for c in title)

    # Collapse multiple underscores/spaces into one
    safe = re.sub(r"[_\s]+", "_", safe).strip("_")

    # Append hash suffix when non-ASCII characters were present
    if any(not c.isascii() for c in title):
        short_hash = hashlib.md5(title.encode("utf-8")).hexdigest()[:6]
        safe = f"{safe}_{short_hash}"

    return safe


def _write_metadata_block(lines, result):
    """Write a structured YAML-style metadata block.

    Fields written (when present and non-empty):
      - source_file  — original document path
      - created_at   — ISO-8601 timestamp of ingestion
      - modified_date — last modification date (if available)
      - tags         — comma-separated list
      - total_words  — word count
      - sections     — numbered heading outline with levels
    """
    metadata = result.get("metadata", {})
    source_file = metadata.get("source_file", "")
    created_at = metadata.get("created_at", "")
    modified_date = metadata.get("modified_date", None)
    tags = result.get("tags", metadata.get("tags", []))
    total_words = metadata.get("total_words")
    sections_list = metadata.get("sections", [])

    meta_lines: list[str] = []

    if source_file:
        meta_lines.append(f"- **Source:** {source_file}")
    if created_at:
        meta_lines.append(f"- **Created:** {created_at}")
    if modified_date:
        meta_lines.append(f"- **Modified:** {modified_date}")

    # Tags and word count on one line (compact)
    compact_parts: list[str] = []
    if tags:
        compact_parts.append(", ".join(tags))
    if total_words:
        compact_parts.append(str(total_words))
    if compact_parts:
        meta_lines.append("- **Tags / Words:** " + " | ".join(compact_parts))

    # Section outline (numbered)
    if sections_list:
        section_items = []
        for s in sections_list:
            level = s.get("level", 2)
            indent = "  " * (level - 2)  # level-2 is base, no indent
            prefix = f"{s.get('number', '')} {indent}" if s.get("number") else indent
            section_items.append(f"{prefix}- **{s['title']}**")
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
