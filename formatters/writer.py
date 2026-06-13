"""Write formatted results to markdown files."""


import os
from datetime import datetime, timezone
from typing import Any, Dict

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
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in title)
    file_path = os.path.join(output_dir, f"{safe_name}.md")

    metadata = result.get("metadata", {})
    
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    
    tags = result.get("tags", [])
    if tags:
        lines.append("**Tags:** " + ", ".join(tags))
    
    total_words = metadata.get("total_words")
    sections = metadata.get("sections", [])
    
    parts = []
    if total_words:
        parts.append(f"**Words:** {total_words}")
    if sections:
        parts.append(f"Sections: {len(sections)}")
    if parts:
        lines.append(" | ".join(parts))

    # Write body content with proper section headers
    body = result.get("body", "")
    if body and isinstance(body, str) and body.strip():
        _write_body_with_sections(lines, body, sections)

    md_content = "\n\n".join(lines).rstrip() + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return file_path


def _render_section_path(path):
    """Render a markdown header from the section path.
    
    Uses ## for single-level sections, ### for nested ones.
    Filters out empty or invalid entries before rendering.
    """
    if not path:
        return ""
    
    # Clean up section names: skip empty entries
    cleaned = []
    for s in path:
        clean = s.strip()
        if not clean:
            continue
        cleaned.append(clean)
    
    if not cleaned:
        return ""
    
    # H2 for single-level, H3+ for nested (H1 is reserved for document title)
    prefix = '#' * (len(cleaned) + 1)
    return "\n\n".join(f"{prefix} {s}" for s in cleaned)


def _write_body_with_sections(lines, body: str, sections: list):
    """Write body text with section headers derived from metadata."""
    if not sections or len(sections) < 2:
        # No clear hierarchy — just write the full body as-is (already has proper markdown formatting)
        lines.append("")
        lines.append(body.strip())
        return

    import re
    
    # Parse actual headers from body text to find offsets and structure
    header_pattern = r'^(#{1,6})\s+(.+)$'
    matches = [(m.group(2).strip(), len(m.group(1)), m.start()) 
               for m in re.finditer(header_pattern, body, re.MULTILINE)]

    if not matches:
        # Body has no headers — write as-is (already formatted)
        lines.append("")
        lines.append(body.strip())
        return

    # The body is already properly formatted markdown from MarkItDown.
    # Just append it directly without re-parsing headers.
    # If LLM output was clean, the body has its own ##/### hierarchy already.
    lines.append("")
    lines.append(body.strip())


def format_md(result):
    """Format result into markdown string (no file write)."""
    os.makedirs("/tmp/md_format_output", exist_ok=True)
    path = write_to_md(result, "/tmp/md_format_output")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


__all__ = ["format_md", "write_to_md"]
