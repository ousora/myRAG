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

    # Metadata block as bullet list — avoids table syntax issues with "|"
    _write_metadata_block(lines, metadata)

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


def _write_metadata_block(lines, metadata):
    """Write a metadata block as bullet list with proper blank-line separation."""
    tags = metadata.get("tags", [])
    total_words = metadata.get("total_words")
    sections_list = metadata.get("sections", [])

    meta_lines = []
    if tags:
        meta_lines.append(f"- **Tags:** {', '.join(tags)}")

    parts = []
    if total_words:
        parts.append(str(total_words))
    if sections_list:
        parts += [s["title"] for s in sections_list]

    if parts:
        meta_lines.append("- **Words:** " + ", ".join(parts))

    # Blank line before and after metadata block (markdown paragraph separation)
    if meta_lines:
        lines.append("")
        lines.extend(meta_lines)
        lines.append("")


def _write_body_with_sections(lines, body: str, sections: list):
    """Write body text with section headers derived from parsed markdown headings.

    Splits the body by actual ``#``/``##``/``###`` headings found in the text,
    then inserts a blank line before each section for proper rendering.
    Falls back to writing the full body as-is when no structure is detected.
    """
    if not sections or len(sections) < 2:
        # No clear hierarchy — write the full body as-is (already has proper markdown formatting)
        lines.append("")
        lines.append(body.strip())
        return

    header_pattern = r'^(#{1,6})\s+(.+)$'
    matches = [(m.group(2).strip(), len(m.group(1)), m.start(), m.end())
               for m in re.finditer(header_pattern, body, re.MULTILINE)]

    if not matches:
        # Body has no headers — write as-is (already formatted)
        lines.append("")
        lines.append(body.strip())
        return

    # Split body into sections by header positions and render each section
    prev_end = 0
    for title, level, start, end in matches:
        if start > prev_end:
            text = body[prev_end:start].strip()
            if text:
                lines.append("")
                lines.append(text)

        # Render header with proper blank-line separation
        prefix = "#" * level
        lines.append(f"\n{prefix} {title}")
        prev_end = end

    # Remaining content after last header
    if len(body) > prev_end:
        text = body[prev_end:].strip()
        if text:
            lines.append("")
            lines.append(text)


def format_md(result):
    """Format result into markdown string (no file write)."""
    os.makedirs("/tmp/md_format_output", exist_ok=True)
    path = write_to_md(result, "/tmp/md_format_output")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


__all__ = ["format_md", "write_to_md"]
