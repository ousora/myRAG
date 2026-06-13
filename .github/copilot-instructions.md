# Project Rules — myrag-pipeline

## Coding Standards

1. **Documentation Language** — Public-facing docs, docstrings, and comments should be in English. Chinese may appear only as supplementary notes for technical concepts where English equivalents are unclear to the team (e.g., RAG 术语解释).

2. **Decoupling** — Functions must be independently testable via unit tests. Keep modules loosely coupled; avoid tight dependencies between parsers, cleaners, and chunkers.

3. **Edge Cases & Error Handling** — Always handle boundary conditions (empty files, malformed input, missing dependencies) gracefully. No silent failures. Use specific `try/except` blocks rather than bare `except:` clauses.

4. **No Hardcoded Values** — Never embed magic numbers or URLs directly in code. Define configurable parameters via YAML config files when multiple values need to be changed together (e.g., embedding endpoints, chunk sizes). For simple per-function parameters, use function arguments with defaults.

5. **Source File Length Limit** — Each source file must not exceed 500 lines. If a module grows beyond this, split it into separate functions/files.

6. **Code Correctness Verification** — Before committing changes:
   - Run unit tests (`pytest`) at minimum; all existing tests must pass
   - Verify modified code imports cleanly and core APIs return expected types
   - Do not commit unverified or broken code

7. **Documentation Updates** — After modifying code, always update the corresponding documentation files (README.md, CHANGELOG.md, `.github/copilot-clause.md`).

8. **Type Hints & Linting** — All public functions must have type annotations (arguments + return types). Run `ruff check` before committing; fix all errors and warnings. Use `mypy` for static type checking when possible.

9. **Logging & Debug Info** — Use the standard `logging` module with appropriate levels (`INFO`, `WARNING`, `ERROR`). Never use `print()` in production code. Log file paths, chunk counts, and parsing results at least at `INFO` level for debugging purposes.

## Testing Requirements

- Each parser module must have unit tests covering its core parsing logic
- Edge cases: empty files, missing dependencies, malformed input
- Integration test for `process_file()` and `process_directory()` end-to-end flow