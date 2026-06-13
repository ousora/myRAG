"""Write formatted results to markdown files."""


import os
from datetime import datetime, timezone
from typing import Any, Dict


def _render_section_path(section_path: list[str]) -> str:
    """Render section path as a hierarchical header.

    Args:
        section_path: e.g., ["CNAPS2概览", "SAPS系统"] → level 3 (H3)
                      ["A", "B", "C"] → level 4 (H4)

    Returns:
        Formatted markdown heading string.
    """
    if not section_path:
        return ""

    # Map path depth to header level (1 = H2, 2 = H3, etc.)
    level = min(len(section_path), 5) + 1  # Clamp at H6
    header_text = " ".join(section_path)

    return f"{'#' * level} {header_text}"


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
    # Safe filename from title — replace spaces/special chars with underscores
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in title)
    file_path = os.path.join(output_dir, f"{safe_name}.md")

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metadata = result.get("metadata", {})

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"**Tags:** {', '.join(result.get('tags', []))} "
        f"| **Words:** {metadata.get('total_words', 0)} "
        f"| **Chunks:** {metadata.get('chunk_count', len(result.get('chunks', [])))}"
    )

    created = metadata.get("created_at", now_str)
    lines.append(f"**Created:** {created}")

    if modified := metadata.get("modified_date"):
        lines.append(f"**Modified:** {modified}")

    sections = metadata.get("sections", [])
    if sections:
        lines.append("")
        for s in sections:
            title_text = s["title"] if isinstance(s, dict) else s
            level_num = s.get("level", 2) if isinstance(s, dict) else 0
            indent = "  " * (level_num - 2)
            lines.append(f"{indent}- {title_text}")

    total_words = metadata.get("total_words", 0)
    chunk_count = metadata.get("chunk_count", len(result.get("chunks", [])))
    lines.append("")
    lines.append(f"**Total words:** {total_words} | **Chunks:** {chunk_count}")

    lines.append("\n---\n")
    for chunk in result.get("chunks", []):
        section_path = chunk.get("section_path", [])
        text = chunk["text"].strip()
        if not text:
            continue
        while text.startswith("\n"):
            text = text[1:]
        while text.endswith("\n"):
            text = text[:-1]

        header = _render_section_path(section_path)
        lines.append(header)
        lines.append("")
        lines.append(text)
        lines.append("")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return file_path


def format_md(result: dict) -> str:
    """Format structured result into markdown string without writing to disk.

    Args:
        result: Output from format_text() — {title, tags, metadata, chunks}

    Returns:
        Markdown text as a single string.
    """
    title = result["title"]
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metadata = result.get("metadata", {})

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"**Tags:** {', '.join(result.get('tags', []))} "
        f"| **Words:** {metadata.get('total_words', 0)} "
        f"| **Chunks:** {metadata.get('chunk_count', len(result.get('chunks', [])))}"
    )

    created = metadata.get("created_at", now_str)
    lines.append(f"**Created:** {created}")

    if modified := metadata.get("modified_date"):
        lines.append(f"**Modified:** {modified}")

    sections = metadata.get("sections", [])
    if sections:
        lines.append("")
        for s in sections:
            title_text = s["title"] if isinstance(s, dict) else s
            level_num = s.get("level", 2) if isinstance(s, dict) else 0
            indent = "  " * (level_num - 2)
            lines.append(f"{indent}- {title_text}")

    total_words = metadata.get("total_words", 0)
    chunk_count = metadata.get("chunk_count", len(result.get("chunks", [])))
    lines.append("")
    lines.append(f"**Total words:** {total_words} | **Chunks:** {chunk_count}")

    lines.append("\n---\n")
    for chunk in result.get("chunks", []):
        section_path = chunk.get("section_path", [])
        text = chunk["text"].strip()
        if not text:
            continue
        while text.startswith("\n"):
            text = text[1:]
        while text.endswith("\n"):
            text = text[:-1]

        header = _render_section_path(section_path)
        lines.append(header)
        lines.append("")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)
