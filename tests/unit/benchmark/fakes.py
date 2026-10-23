"""Shared test doubles for the benchmark package."""

from collections.abc import Sequence

from school_index.embed_index.providers import Vector


class RaisingProvider:
    """A provider whose every embed fails (the credentials-less host).

    Pattern: test double (Strategy) — the blocked-arm tests depend
    on the failure arriving verbatim, the way boto3's
    NoCredentialsError would.
    """

    model_id = "bedrock-titan-embed-v2"
    dim = 8

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        raise RuntimeError("Unable to locate credentials")

    def embed_query(self, text: str) -> Vector:
        raise RuntimeError("Unable to locate credentials")
