"""Entry shim for ``python -m school_index.benchmark``."""

from school_index.benchmark.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
