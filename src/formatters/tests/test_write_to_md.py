import os
import pytest
from formatters.writer import write_to_md

def test_write_to_md_from_doc():
    # Mock result from format_text()
    # Based on the content of doc/AUDIT-2026-06-14.md
    result = {
        "title": "项目审核报告 — myRAG Pipeline",
        "tags": ["audit", "report"],
        "metadata": {
            "source_file": "doc/AUDIT-2026-06-14.md",
            "created_at": "2026-06-14",
            "tags": ["audit", "report"],
            "total_words": 500,
            "sections": [
                {"level": 2, "title": "审核结论"},
                {"level": 2, "title": "修复清单"},
                {"level": 2, "title": "修改文件清单"},
                {"level": 2, "title": "当前架构（清理后）"},
                {"level": 2, "title": "测试状态"}
            ],
        },
        "body": "This is a sample body content for the audit report."
    }

    output_dir = "output/test_audit"
    
    # Run the writer
    path = write_to_md(result, output_dir)
    
    # Assertions
    assert os.path.exists(path)
    assert "项目审核报告" in open(path).read()
    assert "审核结论" in open(path).read()
    assert "修复清单" in open(path).read()

if __name__ == "__main__":
    test_write_to_md_from_doc()
    print("Test passed!")
