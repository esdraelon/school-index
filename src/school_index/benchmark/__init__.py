"""Benchmark: the M5 RAG evaluation for school-index.

Measures the estate's retrieval arms (embedding provider × chunking)
against a versioned, hand-labeled query set built from the live
corpus, and stores every result as records (benchmark.db) so any
number in the summary re-derives from stored data alone.
"""
