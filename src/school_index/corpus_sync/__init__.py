"""Corpus harvester: scheduled pull of resources into the corpus store.

Pattern: Harvester — an anti-corruption layer over the school
adapter's MCP contract (``get_resources``; never its database) —
plus Repository for the ``corpus_items`` / ``sync_state`` schemas.

Incremental by content hash + source identity; vanished resources
become tombstones; per-item failures are stored records, never
silent skips. The sync engine (M1): client.py (the MCP contract),
strategies.py + fetch.py (downloads), store.py (the corpus of
record), harvester.py (the run), cli.py (the composition root).
"""
