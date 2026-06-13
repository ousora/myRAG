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
        for section in sections:
            lines.append(f"- {section}")

    lines.append("\n---\n")
    for chunk in result.get("chunks", []):
        section = chunk.get("section", "General")
        text = chunk["text"].strip()
        if not text:
            continue
        while text.startswith("\n"):
            text = text[1:]
        while text.endswith("\n"):
            text = text[:-1]

        lines.append(f"## {section}")
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
        for section in sections:
            lines.append(f"- {section}")

    lines.append("\n---\n")
    for chunk in result.get("chunks", []):
        section = chunk.get("section", "General")
        text = chunk["text"].strip()
        if not text:
            continue
        while text.startswith("\n"):
            text = text[1:]
        while text.endswith("\n"):
            text = text[:-1]

        lines.append(f"## {section}")
        lines.append("")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)
