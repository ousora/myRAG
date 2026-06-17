# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed
- **Parser lazy loading**: Moved MarkItDown/Trafilatura imports from module level into `__init__`. Parsers now load on first use, allowing the module to be imported even when optional deps are missing (fail-fast in `__init__`). ([src/parsers/dispatcher.py](src/parsers/dispatcher.py))
- **TrafilaturaParser encoding**: Now reads HTML files with UTF-8 → GBK fallback instead of passing filepath directly. Fixes silent parse failures on non-UTF-8 encoded Chinese web pages.

## [Date]

### Added
- Initial release with parser, cleaner, formatter, chunker, embedder, and storage pipeline.
