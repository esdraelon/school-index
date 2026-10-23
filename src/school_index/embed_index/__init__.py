"""Chunking, embedding, and the vector store.

Pattern: Strategy registry for embedding providers (ONNX bge
default; Bedrock as the M5 benchmark arm), injected at the
composition root, plus a Repository over ``chunks`` /
``embed_state`` whose vector persistence sits behind a backend
strategy (sqlite-vec's vec0 when the extension loads, a BLOB +
numpy brute-force fallback otherwise — same interface either way).

The pipeline: chunker.py (structure-aware, deterministic —
``[[page N]]`` markers are the structure), providers.py (the
strategies; ``embed_documents`` vs ``embed_query`` keeps the bge
prefix asymmetry in the API), store.py (the records + backends),
runner.py (delta-only passes keyed on extraction's source_hash;
a model change is an explicit ``--rebuild``, never a silent mix),
cli.py (the composition root: build / fetch-model / search).
Landed in M3.
"""
