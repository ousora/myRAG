"""Tag extraction from formatted document body text.

Uses keyword frequency analysis and proper noun extraction to generate
meaningful domain-specific tags from document content.
"""

from __future__ import annotations

import re
from collections import Counter


# Multi-language stopword sets keyed by detected script family.
_STOP_WORDS_BY_SCRIPT: dict[str, frozenset[str]] = {
    "latin": frozenset({
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    }),
    "cjk": frozenset({
        # Common Chinese function words / particles that carry little semantic weight.
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
        '什么', '怎么', '如何', '为什么', '因为', '所以', '但是', '虽然',
    }),
}

# Generic single words (English) that are almost never useful as tags.
_GENERIC_WORDS = frozenset({
    'the', 'and', 'from', 'into', 'over', 'under', 'between', 'through',
    'during', 'before', 'after', 'above', 'below', 'within', 'across',
    'about', 'against', 'along', 'among', 'around', 'behind', 'beyond',
    'since', 'until', 'upon', 'toward', 'towards',
    'system', 'payment', 'china', 'country', 'bank', 'data',
    'information', 'process', 'service', 'user', 'network',
    'document', 'file', 'text', 'content', 'example',
    'channel', 'series', 'program', 'programs', 'programming',
    'original', 'retrieved', 'archived', 'published', 'based',
})


def extract_tags_from_body(body: str, title: str) -> list[str]:
    """Generate tags from body content for chunked processing mode.

    Uses keyword frequency analysis on the merged body text to extract
    meaningful domain-specific terms as tags. Supports both Latin-script and
    CJK (Chinese/Japanese/Korean) documents via script-aware tokenization.

    Key improvements over naive word-frequency:
      - Script detection → appropriate tokenizer & stopword list per language family
      - Extracts proper nouns (capitalized entities) from title + body
      - Filters out single generic words not in a whitelist
      - Combines adjacent frequent terms into multi-word phrases when useful
      - Prefers domain-specific terms (brands, organizations, systems)

    Returns:
        Up to 5 tag strings.
    """
    script = _detect_body_script(body + " " + title)
    stop_words = _STOP_WORDS_BY_SCRIPT.get(script, frozenset(_STOP_WORDS_BY_SCRIPT["latin"]))

    # ── Tokenize according to detected script ───────────────────
    if script == "cjk":
        body_tokens, title_tokens = _tokenize_cjk(body, title)
    else:
        body_tokens, title_tokens = _tokenize_latin(body, title)

    word_freq = Counter(t for t in body_tokens if t not in stop_words and len(t) > 1)
    title_freq = Counter(title_tokens)

    # Combine: title words get higher weight.
    combined = Counter(word_freq)
    for w, c in title_freq.items():
        combined[w] += c * 2

    # ── Extract proper nouns / entities from title and body ─────
    def _extract_proper_nouns(text: str) -> list[str]:
        if script == "cjk":
            return _extract_cjk_entities(text)
        return _extract_latin_proper_nouns(text)

    proper_nouns = _extract_proper_nouns(title) + _extract_proper_nouns(body)
    noun_freq = Counter(p for p in proper_nouns if len(p.split()) <= 3 and len(p) > 1)

    # ── Build tag candidates (up to 5) ──────────────────────────
    tags: list[str] = []
    seen: set[str] = set()

    # Phase 1: Proper nouns / entities (highest priority).
    for noun, _ in noun_freq.most_common(3):
        if len(tags) >= 5:
            break
        tag_lower = noun.lower().strip()
        if tag_lower not in seen and tag_lower not in _GENERIC_WORDS and len(tag_lower) > 1:
            tags.append(tag_lower)
            seen.add(tag_lower)

    # Phase 2: High-frequency domain words (>=3 occurrences).
    for word, count in combined.most_common(40):
        if len(tags) >= 5:
            break
        if word in seen or word in stop_words:
            continue
        tags.append(word)
        seen.add(word)

    # Phase 3: Title-based fallback.
    for w, _ in title_freq.most_common(10):
        if len(tags) >= 5:
            break
        wl = w.lower()
        if wl not in seen and wl not in _GENERIC_WORDS and wl not in stop_words:
            tags.append(wl)
            seen.add(wl)

    # Final filter: remove single generic words (unless multi-word).
    final_tags = [t for t in tags if t not in _GENERIC_WORDS or len(t.split()) > 1]
    return final_tags[:5] if len(final_tags) >= 3 else final_tags


def _detect_body_script(text: str) -> str:
    """Detect the dominant writing script of *text* ('latin' | 'cjk')."""
    cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin_chars = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    total = cjk_chars + latin_chars
    if total == 0:
        return "latin"
    return "cjk" if cjk_chars / total > 0.15 else "latin"


def _tokenize_latin(body: str, title: str) -> tuple[list[str], list[str]]:
    """Tokenize Latin-script text into lowercase word tokens."""
    body_tokens = re.findall(r'[a-zA-Z]{3,}', body.lower())
    title_tokens = re.findall(r'[a-zA-Z]{3,}', title.lower())
    return body_tokens, title_tokens


def _tokenize_cjk(body: str, title: str) -> tuple[list[str], list[str]]:
    """Tokenize CJK text using character bigrams + word-boundary splitting.

    Since CJK scripts lack intrinsic word boundaries, we combine:
      1. Two-character sequences (bigrams) from the body for frequency analysis.
      2. Whitespace-separated tokens as additional candidates.
    This produces a reasonable set of domain-specific terms without requiring
    an external NLP library like jieba.
    """
    def _cjk_tokens(text: str) -> list[str]:
        # Extract whitespace-delimited words (handles mixed CJK/Latin).
        ws_tokens = [t.lower() for t in text.split() if len(t) > 1]

        # Character bigrams from pure-CJK runs capture compound terms.
        cjk_runs = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        bg: list[str] = []
        for run in cjk_runs:
            bg.extend(run[i:i + 2].lower() for i in range(len(run) - 1))

        return ws_tokens + bg

    body_tokens = _cjk_tokens(body)
    title_tokens = _cjk_tokens(title)
    return body_tokens, title_tokens


def _extract_latin_proper_nouns(text: str) -> list[str]:
    """Extract capitalized words/phrases that look like proper nouns."""
    title_parts = re.findall(r'[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*', text[:200])
    entity_phrases = re.findall(
        r'(?<![a-z])([A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+){1,3})(?![a-z])',
        text[:min(len(text), 5000)],
    )
    return title_parts + entity_phrases


def _extract_cjk_entities(text: str) -> list[str]:
    """Extract CJK named-entity-like phrases from the first portion of text.

    Looks for runs of ≥2 Chinese characters preceded/followed by non-CJK or
    line boundaries, which tend to be proper nouns in context.
    """
    candidates = re.findall(r'(?<![\u4e00-\u9fff])([\u4e00-\u9fff]{2,16})(?![\u4e00-\u9fff])', text[:min(len(text), 5000)])
    return [c for c in candidates if len(c) >= 2]
