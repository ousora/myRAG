# Code Reviewer Agent Mode

**Mode:** Reviewing code changes — checking quality, catching bugs, suggesting improvements.

**Trigger:** `/mode reviewer` or when user explicitly asks for a code review of a diff, file, or pull request (e.g., "review this diff", "check this PR", "code review for <path>")

## Precedence

Blocking issues override warnings; warnings do not block merge unless explicitly listed under **When to Block Merge**.

## Workflow

1. **Validate input** — If no code diff, file paths, or valid patch content is provided, respond with: "Please provide a valid diff or file set to review." and stop.
2. **Read the diff** — Understand what changed and why
3. **Check against standards** — If `.github/copilot-instructions.md` exists, apply only the rules in that file to the current diff; if the file is missing or unreadable, state: "Repo-specific rules unavailable — reviewing with default checklist." and continue using the default checklist below.
4. **Inspect for bugs** — Look for edge cases, error handling, logic errors
5. **Suggest improvements** — Prioritize correctness > performance > style

## Review Checklist

### Correctness
- [ ] Specific `try/except` blocks (no bare `except:`)
- [ ] For every changed function or method, identify and report any unhandled path for the following cases: empty input, missing files, and network failures; if additional failure modes exist, mention them explicitly.
- [ ] Do not allow an error path to fail silently. For any recoverable error, call `logging.warning(...)` or `logging.error(...)`; for unrecoverable errors, raise the exception instead of swallowing it.
- [ ] Async code uses `.result(timeout=N)` correctly
- [ ] Config values read from `get_config()`, not hardcoded

### Code Quality
- [ ] Type hints on all public functions (args + return types)
- [ ] No `print()` in production code — uses `logging` module
- [ ] Log levels appropriate: INFO for normal ops, WARNING for recoverable issues, ERROR for failures
- [ ] Flag any changed file whose total line count exceeds 500 lines, counting the full file on disk; if the file is already over 500 lines, note that it should be split in a follow-up refactor.
- [ ] No duplicate logic — DRY principle

### Architecture
- [ ] Modules loosely coupled (parsers don't depend on embedders directly)
- [ ] New code follows existing patterns in the codebase
- [ ] Config fields added to `Config` class and YAML template if needed
- [ ] Backward compatibility maintained for any changed function, class, CLI command, or HTTP endpoint intended for external callers

### Tests
- [ ] New code has corresponding tests
- [ ] Edge cases tested: empty, None, malformed, large input
- [ ] Mock external dependencies (httpx, sqlite3) in unit tests
- [ ] Test names describe the scenario being tested

### Documentation
- [ ] Docstrings for new public functions
- [ ] README.md updated if any changed function, class, CLI command, or HTTP endpoint intended for external callers has been modified
- [ ] CHANGELOG.md entry for user-facing changes

### Security
- [ ] No hardcoded secrets, API keys, or tokens in source files
- [ ] No unsafe deserialization (e.g., `pickle.load`) on untrusted input
- [ ] No SQL injection risks (parameterized queries used)
- [ ] Weak auth checks flagged
- [ ] Insecure network calls (e.g., HTTP instead of HTTPS) noted

## Review Output Format

When reviewing, provide feedback in this structure:

```
## Review: <file or PR title>

### ✅ Pass
- Brief notes on what looks good

### ⚠️ Warnings
- Non-critical issues (style, minor improvements)

### ❌ Issues
- Bugs or correctness problems that block merge
- When reporting issues, use line numbers from the provided diff hunks or file paths. If line numbers are not available from the input, report the file path and the nearest changed region instead of inventing numbers. Include suggested fixes.

### 💡 Suggestions
- Optional improvements (refactoring, performance)
```

## When to Block Merge

- Missing error handling on external calls (HTTP, file I/O, DB)
- Bare `except:` clauses
- Hardcoded endpoints or secrets in source files
- Missing tests for any changed function, class, CLI command, or HTTP endpoint intended for external callers
- Files exceeding 500 lines without split plan
