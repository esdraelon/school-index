"""Shared test doubles for embed_index: a deterministic fake provider.

The fake's vectors are a normalized hash projection of the text —
deterministic across runs and machines, and different texts get
different vectors, which is all the pipeline tests need. Retrieval
*quality* is the real-model test's job (and M5's benchmark).
"""

import hashlib
import math
from collections.abc import Sequence

from school_index.embed_index.providers import Vector


class FakeProvider:
    """Deterministic hash provider. Pattern: test double (Strategy)."""

    model_id = "fake-hash"
    dim = 16

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [_vector(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return _vector(text)


def _vector(text: str) -> Vector:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [byte / 127.5 - 1.0 for byte in digest[: FakeProvider.dim]]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return tuple(x / norm for x in raw)
