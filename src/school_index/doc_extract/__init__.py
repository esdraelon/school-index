"""Text extraction from harvested files (Slides, Docs, Sheets, PDF).

Pattern: Strategy registry — a Mapping from format key to Extractor
strategy, injected at the composition root (no if/else selection,
per the coding standard).

Per-item failure containment: one bad file never fails a run, and
the failure is stored as a record. The registry lands in M2
(shell only in M0).
"""
