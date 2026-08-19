"""Allow ``python -m src`` to invoke the CLI entry point."""

import sys

from pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main())
