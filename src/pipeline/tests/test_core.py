"""Tests for pipeline.core reference-section stripping."""

from pipeline.core import _build_doc_summary
from pipeline.markdown_utils import is_reference_title, strip_reference_sections


def test_build_doc_summary_short_body():
    s = _build_doc_summary("Title", ["t1"], "short body")
    assert "Title: Title" in s
    assert "Tags: t1" in s
    assert "short body" in s


def test_build_doc_summary_head_tail():
    body = "A" * 900 + "\nEND\n" + "B" * 500
    s = _build_doc_summary("Doc", [], body)
    # Head preserved, tail appended (separated by a marker) for long bodies.
    assert "A" * 50 in s
    assert "B" * 50 in s
    assert "..." in s


def test_build_doc_summary_empty():
    assert _build_doc_summary("", [], "").startswith("Title:")


def test_is_reference_title_english():
    assert is_reference_title("References")
    assert is_reference_title("Further reading")
    assert is_reference_title("Bibliography")
    assert not is_reference_title("Introduction")
    assert not is_reference_title("Architecture")


def test_is_reference_title_cjk():
    assert is_reference_title("参考文献")
    assert is_reference_title("參考")
    assert is_reference_title("引用文献")
    assert not is_reference_title("架构")
    assert not is_reference_title("检索增強生成")


def test_strip_reference_sections_removes_block():
    md = (
        "# Doc\n\n"
        "## Intro\n\nSome intro text.\n\n"
        "## References\n\n"
        "[1] Author. Title. 2020.\n"
        "[2] Another. Work. 2021.\n\n"
        "## Conclusion\n\nFinal words."
    )
    out = strip_reference_sections(md)
    assert "References" not in out
    assert "[1] Author" not in out
    assert "## Intro" in out
    assert "Some intro text." in out
    assert "## Conclusion" in out
    assert "Final words." in out


def test_strip_reference_sections_cjk():
    md = (
        "# 文档\n\n"
        "## 概述\n\n正文内容。\n\n"
        "## 參考\n\n[1] 作者. 文献. \n\n"
        "## 總結\n\n总结文字。"
    )
    out = strip_reference_sections(md)
    assert "參考" not in out
    assert "[1] 作者" not in out
    assert "总结文字。" in out


def test_strip_reference_sections_no_refs_unchanged():
    md = "# Doc\n\n## A\n\nalpha.\n\n## B\n\nbeta."
    assert strip_reference_sections(md) == md


def test_strip_reference_sections_last_section():
    md = "# Doc\n\n## Body\n\ncontent.\n\n## References\n\n[1] ref one.\n[2] ref two."
    out = strip_reference_sections(md)
    assert "ref one" not in out
    assert "ref two" not in out
    assert "content." in out
