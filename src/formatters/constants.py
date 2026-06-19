"""JSON schemas for LLM response_format (llama.cpp / OpenAI-compatible).

Used by call_llm() to enforce output structure via the model's native JSON mode.
See: https://platform.openai.com/docs/guides/structured-outputs/json-schema-support
     llama-cpp-python supports this via the 'response_format' parameter.

Note for llama.cpp users:
  - `maxLength` on string fields causes "peg-native format" errors when the value is large.
    Use a reasonable limit (e.g., 32768) or omit it entirely; rely on API max_tokens instead.
"""

# ── Formatter (single-shot) schema ────────────────────────────────────────
# Matches the dict shape returned by format_text_single / _format_text_chunked_merged.
FORMATTER_SCHEMA = {
    "type": "object",
    "required": ["title", "tags", "metadata", "body"],
    "properties": {
        "title": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
        },
        "metadata": {
            "type": "object",
            "properties": {
                "source_type": {"type": "string"},
                "total_words": {"type": "integer"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["level", "title"],
                        "properties": {
                            "level": {"type": "integer"},
                            "title": {"type": "string"},
                        },
                    },
                },
                "created_at": {"type": "string"},
                "modified_date": {"type": ["string", "null"]},
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "type"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["PERSON", "ORG", "PRODUCT", "LOCATION", "CONCEPT"],
                            },
                        },
                    },
                },
            },
        },
        # body is the largest field — must be a string (may contain quotes, newlines).
        # No maxLength here; rely on API max_tokens to bound output size.
        "body": {"type": "string"},
    },
}

# ── Chunked (multi-chunk) schema ────────────────────────────────────────
CHUNKED_SCHEMA = {
    "type": "object",
    "required": ["part_md", "summary"],
    "properties": {
        "part_md": {"type": "string"},
        "summary": {"type": "string"},
    },
}
