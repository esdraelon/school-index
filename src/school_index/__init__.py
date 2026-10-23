"""School Index: searchable corpus over harvested school material.

Pattern: a composed pipeline of four subpackages — ``corpus_sync``
harvests resources through the school adapter's MCP contract,
``doc_extract`` turns files into text, ``embed_index`` chunks,
embeds, and stores vectors, and ``index_mcp`` serves search
read-only over the store. Each stage names its own pattern in its
subpackage docstring (per the coding standard).

Design: school-assistant/docs/INDEX-DESIGN.md.
"""
