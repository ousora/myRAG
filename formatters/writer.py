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

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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
    lines.append("")

    # Build a section header index from metadata.sections (NOT from chunks)
    if sections:
        lines.append("**Sections:**")
        for s in sections:
            level = min(s.get("level", 2), 6) - 1  # Convert to markdown heading (# + N-1)
            title_text = s.get("title", "")
            indent = "  " * max(level, 0)
            marker = "#" * (level + 1) if level >= 1 else ""
            lines.append(f"{indent}- {marker} {title_text}")
        lines.append("")

    # Write chunks as body — minimal markers for readability only when needed
    chunks = result.get("chunks", [])
    if chunks:
        prev_section = None
        lines.append("")
        for i, c in enumerate(chunks):
            text = c.get("text", "") or ""
            sp = tuple(c.get("section_path") or ["General"])
            
            # Only add a marker when section changes (not every chunk)
            if prev_section and sp != prev_section:
                path_str = " / ".join(sp) if isinstance(sp, list) else str(sp)
                lines.append("")
                lines.append(f"---")  # Section separator instead of header
                lines.append("")
            
            lines.append(text.strip())
            prev_section = sp
        lines.append("")

    md_content = "\n".join(lines)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return file_path


def format_md(result: dict) -> str:
    """Format result into markdown string (no file write)."""
    os.makedirs("/tmp/md_format_output", exist_ok=True)
    path = write_to_md(result, "/tmp/md_format_output")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Re-export for convenience from formatters module
__all__ = ["format_md", "write_to_md"]