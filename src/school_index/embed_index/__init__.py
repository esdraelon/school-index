"""Chunking, embedding, and the vector store.

Pattern: Strategy registry for embedding providers (ONNX bge
default; Bedrock as the M5 benchmark arm), injected at the
composition root, plus Repository over sqlite-vec for the
``chunks`` / ``embeddings`` schemas.

Embeddings are keyed by chunk + model_id; only changed chunks
re-embed, and a model change is an explicit full re-embed op.
Chunking and embedding land in M3 (shell only in M0).
"""
