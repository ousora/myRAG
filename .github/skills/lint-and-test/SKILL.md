# Lint & Test Skill

Run linting and tests before committing changes.

## When to use

- After modifying any source code in `src/`
- Before suggesting a commit
- When the user says "check", "lint", "test", or "verify"

## Steps

1. **Lint**
   ```bash
   uv run ruff check .
   ```
   Fix all errors and warnings. If there are no errors, proceed.

2. **Test**
   ```bash
   uv run pytest -v
   ```
   All 22 tests must pass (chunkers 8 + formatters 9 + cleaners 5).

3. **Report**
   - If both pass: "Lint and tests passed ✓"
   - If lint fails: show errors, offer to fix them
   - If tests fail: show failures with line numbers, identify which modules are affected

## Notes

- Tests live in `src/chunkers/tests/`, `src/formatters/tests/`, `src/cleaners/tests/`
- The project uses `uv` for dependency management — always use `uv run` prefix
- If a new module is added, ensure it has corresponding tests before committing
