"""Write formatted results to markdown files."""


import os
from datetime import datetime, timezone
from typing import Any, Dict

def write_to_md(result: dict, output_dir: str) -> str:
    """Format structured result into markdown and save it.

    Args:
        result: Output from format_text() — {title, tags, metadata, chunks}
        output_dir: Directory to save the .md file (created if needed).

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    title = result["title"]
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in title)
    file_path = os.path.join(output_dir, f"{safe_name}.md")

    metadata = result.get("metadata", {})
    sections = metadata.get("sections", [])  # [{level, title}, ...] from LLM
    
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    
    tags = result.get("tags", [])
    if tags:
        lines.append("**Tags:** " + ", ".join(tags))
    
    total_words = metadata.get("total_words")
    chunk_count = metadata.get("chunk_count")
    if total_words or chunk_count:
        parts = []
        if total_words:
            parts.append(f"**Words:** {total_words}")
        if chunk_count:
            parts.append(f"**Chunks:** {chunk_count}")
        lines.append(" | ".join(parts))
    
    # Build a hierarchical section TOC from metadata.sections (NOT from chunks)
    if sections:
        lines.append("")
        for s in sections:
            # LLM formatter outputs {level: N} where N = heading number (2=H2/##, 3=H3/###)
            level = min(s.get("level", 2), 6)
            title_text = s.get("title", "")
            indent = "  " * max(level - 1, 0)
            lines.append(f"{indent}{'#' * level} {title_text}")
    
    lines.append("")

    # Write chunks as body content — NO section headers in text
    chunks = result.get("chunks", [])
    if chunks:
        prev_section = None
        for c in chunks:
            text = c.get("text", "") or ""
            
            # Clean text: strip any leading section header that LLM may have included
            cleaned_text = _strip_section_header(text.strip()) if isinstance(text, str) else text
            
            lines.append(cleaned_text)
    
    md_content = "\n".join(lines) + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return file_path


def _strip_section_header(text: str) -> str:
    """Remove leading section header/anchor info from chunk text.

    LLM formatter may include section title info at the start of chunk text
    for vector search anchoring (e.g., "第一篇 CNAPS 是..." or "导图\n...").
    
    Returns cleaned text with the leading anchor removed.
    """
    import re
    
    # Strip article markers like 第一篇/第二篇 etc. followed by newline
    text = re.sub(r'^[第]([一二三四五六七八九十百]+)\s*\n', '', text)
    
    # Pattern: markdown header at start of chunk (## Header\n or ### Header\n)
    text = re.sub(r'^(#{1,6})\s+[^\n]+\n+', '', text)
    
    # Pattern: "导图" or similar single-word section anchors followed by newline
    text = re.sub(r'^[导]\w+\s*\n', '', text)
    
    return text.strip()


def format_md(result: dict) -> str:
    """Format result into markdown string (no file write)."""
    os.makedirs("/tmp/md_format_output", exist_ok=True)
    path = write_to_md(result, "/tmp/md_format_output")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


__all__ = ["format_md", "write_to_md"]
