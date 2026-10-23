"""Text extraction over the harvested corpus.

Pattern: Strategy registry — a Mapping from format key to Extractor
strategy, injected at the composition root (no if/else selection,
per the coding standard) — plus a per-stage Repository over this
stage's own tables (``item_text`` / ``extraction_state``) in the
shared corpus database.

Every corpus item gets an outcome record: extracted (text stored),
empty (valid file, no text layer), failed (error stored), or
not-extractable (no strategy for its format, or no blob by design).
Re-runs are delta-only — an item is re-extracted only when its
content hash changed or its last outcome was a failure. The
pipeline: extractors.py (the strategies), store.py (the records),
runner.py (the run), cli.py (the composition root). The stage
landed in M2 with exactly one real format in the corpus — PDF, via
pypdf — and the registry is where later formats slot in.
"""
