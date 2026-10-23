"""Search MCP: passage search, topic -> resources, worked examples.

Pattern: FastMCP application — an Adapter over the corpus store,
read-only. Every result carries provenance (item, source, and the
original deep link) so consumers cite records, not recollections.

Served per the gateway deployment standard: MCP at the app root,
public host allowlisted, uvicorn proxy headers on, and its own
path-scoped gateway token (client ``index-mcp``).

Modules: ``domain`` (Passage / TopicResource records), ``store``
(the SELECT-only IndexReadStore), ``queries`` (SearchService — the
ranking logic), ``serializers`` (wire shapes), ``server`` (the
FastMCP app + bearer middleware wiring), ``auth`` (the middleware),
``config`` (environment -> ServerConfig), ``cli`` (the composition
root: config, registry, census check, uvicorn).
"""
