"""``python -m school_index.index_mcp`` — delegates to the CLI root."""

from school_index.index_mcp.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
