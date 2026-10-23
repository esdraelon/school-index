"""Search MCP: passage search, topic -> resources, worked examples.

Pattern: FastMCP application — an Adapter over the corpus store,
read-only. Every result carries provenance (item, source, and the
original deep link) so consumers cite records, not recollections.

Served per the gateway deployment standard: MCP at the app root,
public host allowlisted, uvicorn proxy headers on, and its own
path-scoped gateway token (client ``index-mcp``). The server lands
in M4 (shell only in M0).
"""
