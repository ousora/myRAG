# Project Rules — myrag-pipeline

> **Full agent instructions are in [AGENTS.md](AGENTS.md)** — architecture, key directories, build commands, conventions, and pitfalls. This file contains the coding standards only.

## Coding Standards

1. **English Only** — All code comments, docstrings, documentation files (README, CHANGELOG, etc.), and commit messages must be written in English exclusively. No Chinese or other languages permitted.

2. **No Emojis** — Do not use emojis anywhere: code, comments, documentation, commit messages, or chat responses.

3. **Professional Tone** — Maintain a professional tone throughout all code and documentation. Follow common coding standards (PEP 8 for Python, SOLID principles, DRY/KISS). Avoid casual language, slang, or informal expressions.

4. **Lists Over Tables** — Prefer bullet-point lists over tables in all documents. Keep document content clear, concise, and well-organized with proper headings and structure.

5. **Decoupling** — Functions must be independently testable via unit tests. Keep modules loosely coupled; avoid tight dependencies between parsers, cleaners, and chunkers.

6. **Edge Cases & Error Handling** — Always handle boundary conditions (empty files, malformed input, missing dependencies) gracefully. No silent failures. Use specific `try/except` blocks rather than bare `except:` clauses.

7. **No Hardcoded Values** — Never embed magic numbers or URLs directly in code. Define configurable parameters via YAML config files when multiple values need to be changed together (e.g., embedding endpoints, chunk sizes). For simple per-function parameters, use function arguments with defaults.

8. **Source File Length Limit** — Each source file must not exceed 500 lines. If a module grows beyond this, split it into separate functions/files.

9. **Code Correctness Verification** — Before committing changes:
   - Run unit tests (`pytest`) at minimum; all existing tests must pass
   - Verify modified code imports cleanly and core APIs return expected types
   - Do not commit unverified or broken code

10. **Documentation Updates** — After modifying code, always update the corresponding documentation files (README.md, CHANGELOG.md, `.github/copilot-clause.md`).

11. **Type Hints & Linting** — All public functions must have type annotations (arguments + return types). Run `ruff check` before committing; fix all errors and warnings. Use `mypy` for static type checking when possible.

12. **Logging & Debug Info** — Use the standard `logging` module with appropriate levels (`INFO`, `WARNING`, `ERROR`). Never use `print()` in production code. Log file paths, chunk counts, and parsing results at least at `INFO` level for debugging purposes.

## Testing Requirements

- Each parser module must have unit tests covering its core parsing logic
- Edge cases: empty files, missing dependencies, malformed input
- Integration test for `process_file()` and `process_directory()` end-to-end flow