"""``python -m school_index.corpus_sync`` — delegates to the CLI root."""

from school_index.corpus_sync.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
