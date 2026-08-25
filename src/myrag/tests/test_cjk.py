"""Tests for CJK segmentation used by the FTS index (myrag.cjk)."""

from myrag.cjk import contains_cjk, count_cjk, segment_for_fts


class TestSegmentForFts:
    def test_cjk_run_gets_per_char_boundaries(self):
        assert segment_for_fts("风险管控") == "风 险 管 控"

    def test_cjk_latin_boundary_is_split(self):
        # unicode61 would otherwise merge "AI系统" into one token
        assert segment_for_fts("AI系统v2") == "AI 系 统 v2"

    def test_latin_text_untouched(self):
        text = "hello world v2.0"
        assert segment_for_fts(text) == text

    def test_fullwidth_punctuation_not_cjk(self):
        # unicode61 already token-splits on punctuation; no CJK-boundary rules apply
        assert segment_for_fts("你好，世界。") == "你 好，世 界。"

    def test_empty(self):
        assert segment_for_fts("") == ""


class TestCjkHelpers:
    def test_contains_and_count(self):
        assert contains_cjk("中文")
        assert not contains_cjk("latin only")
        assert count_cjk("中文abc") == 2
