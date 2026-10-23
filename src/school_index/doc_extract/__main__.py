"""``python -m school_index.doc_extract`` — delegates to the CLI root."""

from school_index.doc_extract.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
