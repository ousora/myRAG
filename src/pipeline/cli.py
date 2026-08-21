"""CLI entry point for the RAG pipeline."""

import argparse
import json
import logging
import logging.handlers
import sys
from pathlib import Path

from config import get_config_lazy as _get_config

logger = logging.getLogger(__name__)


def _cli_print(*args, **kwargs):
    """Print to stdout so it's separated from logging (which goes to stderr)."""
    print(*args, **kwargs)


def main() -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    parser = argparse.ArgumentParser(description="RAG data cleanup pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # process_file command (traditional)
    file_parser = subparsers.add_parser("process-file", help="Process a single file (traditional)")
    file_parser.add_argument("input", help="file to process")
    file_parser.add_argument("--chunk-size", type=int, default=1024)

    # process_directory command (batch traditional)
    dir_parser = subparsers.add_parser("process-directory", help="Process all files in directory")
    dir_parser.add_argument("input", help="directory to process")
    dir_parser.add_argument("--chunk-size", type=int, default=1024)

    # ingest command: .md → chunk → embed → sqlite-vec
    ingest_parser = subparsers.add_parser("ingest", help="Chunk .md file and store to sqlite-vec")
    ingest_parser.add_argument("input", help=".md file to ingest")
    ingest_parser.add_argument("--store", required=True, help="Path to sqlite-vec database")
    ingest_parser.add_argument("--doc-id", default="doc_0", help="Document ID for storage")
    ingest_parser.add_argument("--chunk-size", type=int, default=1024)

    # process command: .md → sqlite-vec (two-step: generate .md, then ingest)
    process_parser = subparsers.add_parser("process", help="Generate .md and store to sqlite-vec")
    process_parser.add_argument("input", help="file to process")
    process_parser.add_argument("--store", required=True, help="Path to sqlite-vec database")
    process_parser.add_argument("--doc-id", default="doc_0", help="Document ID for storage")
    process_parser.add_argument("--output-dir", default="./output/", help="Output directory for .md files")
    process_parser.add_argument("--chunk-size", type=int, default=1024)
    process_parser.add_argument("--no-llm", action="store_true", help="Skip LLM formatting; write parser output directly")

    # md command (generate markdown output)
    md_parser = subparsers.add_parser("md", help="Generate structured markdown output only")
    md_parser.add_argument("input", help="file to process")
    md_parser.add_argument("--output-dir", default="./output/", help="Output directory for .md files")
    md_parser.add_argument("--no-llm", action="store_true", help="Skip LLM formatting; write parser output directly")

    # hybrid command
    hybrid_parser = subparsers.add_parser("hybrid", help="Process file with hybrid indexing")
    hybrid_parser.add_argument("input", help="file to process")
    hybrid_parser.add_argument("--store", required=True, help="Path to sqlite-vec database")
    hybrid_parser.add_argument("--doc-id", default="doc_0", help="Document ID for storage")

    args = parser.parse_args()

    # Setup logging: console + file
    log_dir = Path("logs")
    if log_dir.is_file():
        _msg = f"'{log_dir}' exists as a file, not a directory"
        raise OSError(_msg)
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "pipeline.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(console)

    cfg = _get_config()

    # Log loaded configuration for debugging
    root_logger.info("Config: log_max_bytes=%d, chunk_threshold_chars=%d",
                     cfg.log_max_bytes, cfg.chunk_threshold_chars)
    root_logger.info("Config: format_timeout=%ds, chunk_timeout=%ds",
                     cfg.format_timeout, cfg.chunk_timeout)

    # File handler (append, configurable max size, keep 3 backups)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=cfg.log_max_bytes, backupCount=3, encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(fh)

    logger = logging.getLogger(__name__)
    logger.info("Logging to %s", log_file.resolve())

    if args.command == "process-file":
        from pipeline.core import process_file
        chunks = process_file(args.input, chunk_size=args.chunk_size)
        _cli_print(json.dumps(chunks, indent=2, ensure_ascii=False))

    elif args.command == "process-directory":
        from pipeline.core import process_directory
        chunks = process_directory(args.input, chunk_size=args.chunk_size)
        _cli_print(json.dumps(chunks, indent=2, ensure_ascii=False))

    elif args.command == "hybrid":
        from pipeline.core import process_file_hybrid
        result = process_file_hybrid(args.input, doc_id=args.doc_id, store_path=args.store)
        _cli_print(f"Chunks: {len(result['chunks'])}")
        if result.get("db_path"):
            _cli_print(f"DB:     {result['db_path']}")
        _cli_print("Document index created")
        if result.get("format_result"):
            _cli_print(f"Title: {result['format_result'].get('title', 'N/A')}")

    elif args.command == "ingest":
        from pipeline.ingest import _ingest_markdown
        db_path = _ingest_markdown(
            args.input, store_path=args.store,
            doc_id=args.doc_id, chunk_size=args.chunk_size,
        )
        _cli_print(f"DB:     {db_path}")

    elif args.command == "process":
        from pipeline.core import process_file_with_md
        # Step 1: generate .md
        md_path = process_file_with_md(args.input, output_dir=args.output_dir,
                                         use_llm=not args.no_llm)
        if not md_path:
            logger.error("Failed to generate markdown for %s", args.input)
            return 1

        # Step 2: ingest from .md
        from pipeline.ingest import _ingest_markdown
        db_path = _ingest_markdown(
            md_path, store_path=args.store,
            doc_id=args.doc_id, chunk_size=args.chunk_size,
        )
        _cli_print(f"Written to: {md_path}")
        _cli_print(f"DB:         {db_path}")

    elif args.command == "md":
        from pipeline.core import process_file_with_md
        path = process_file_with_md(args.input, output_dir=args.output_dir,
                                      use_llm=not args.no_llm)
        if path:
            _cli_print(f"Written to: {path}")
        else:
            logger.error("Failed to generate markdown for %s", args.input)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
