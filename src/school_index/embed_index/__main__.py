"""``python -m school_index.embed_index`` — delegates to the CLI root."""

from school_index.embed_index.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
