import os
import threading

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
    assert "项目审核报告" in open(path, encoding="utf-8").read()
    assert "审核结论" in open(path, encoding="utf-8").read()
    assert "修复清单" in open(path, encoding="utf-8").read()


def _make_result(title: str, body: str) -> dict:
    return {"title": title, "tags": [], "metadata": {}, "body": body}


def test_concurrent_same_title_no_clobber(tmp_path):
    """Concurrent same-title writes must each survive (collision suffix, not overwrite)."""
    n = 8
    results = [(_make_result("Same Title", f"unique body {i} " + "x" * 200)) for i in range(n)]
    paths: list[list[str]] = [[] for _ in range(n)]
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            paths[i].append(write_to_md(results[i], tmp_path))
        except Exception as exc:  # noqa: BLE001 — collected below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Worker failures: {errors}"
    written = [p[0] for p in paths]
    # Every thread's content must still be readable at its returned path —
    # no output lost to a racing overwrite.
    contents = {p: open(p, encoding="utf-8").read() for p in written}
    for i, p in enumerate(written):
        assert f"unique body {i}" in contents[p], f"Thread {i}'s file was clobbered"
    # Distinct contents got distinct files.
    assert len(set(written)) == n


if __name__ == "__main__":
    test_write_to_md_from_doc()
    print("Test passed!")
