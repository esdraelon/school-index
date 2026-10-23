"""Benchmark records: the labeled set, raw outcomes, stored runs.

Pattern: frozen dataclasses (keyword-only construction). The split
that keeps the benchmark honest:

- ``QueryCase`` / ``QuerySet`` — the labeled sample, versioned as
  data (benchmark/queries.json), never as code.
- ``QueryOutcome`` — the RAW per-query evidence (ranked item ids +
  scores, exactly as the arm returned them, plus latencies). Runs
  store outcomes, not scores: every metric is computed from these
  records by ``benchmark.metrics``, so the report re-derives from
  the store alone and a metric-definition fix never needs a re-run.
- ``RunRecord`` — one arm's pass over one query-set version:
  completed (with outcomes) or blocked (with the reason verbatim).
  Blocked is a result, recorded with the same care as a number.
"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class QueryCase:
    """One labeled query.

    ``expected_item_ids`` names the corpus items that SHOULD
    surface (1–3; empty = the topic is genuinely absent from the
    corpus and the query measures graceful behavior instead).
    ``course`` is the hint a tutor would pass, or None.
    """

    id: str
    query: str
    course: str | None
    expected_item_ids: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True, kw_only=True)
class QuerySet:
    """A versioned labeled set + the census date its labels came from."""

    set_version: str
    queries: tuple[QueryCase, ...]
    corpus_census_date: str | None = None


@dataclass(frozen=True, kw_only=True)
class ResourceRank:
    """One item in a resources_for_topic ranking, in rank order."""

    item_id: str
    best_score: float
    rank_score: float


@dataclass(frozen=True, kw_only=True)
class RankedItem:
    """One item in an item ranking: its best (first-seen) score."""

    item_id: str
    score: float


@dataclass(frozen=True, kw_only=True)
class QueryOutcome:
    """Raw evidence for one query under one arm.

    ``resources`` is the item ranking resources_for_topic returned
    (rank order); ``passages`` is the item ranking derived from
    search_passages (first occurrence per item, in chunk-rank
    order); ``top_chunk_item_id`` is the owner of the single best
    chunk. ``expected_item_ids`` is snapshotted from the query set
    so the record stands alone.
    """

    query_id: str
    expected_item_ids: tuple[str, ...]
    resources: tuple[ResourceRank, ...]
    passages: tuple[RankedItem, ...]
    top_chunk_item_id: str | None
    latency_resources_ms: float
    latency_passages_ms: float


@dataclass(frozen=True, kw_only=True)
class RunRecord:
    """One arm's pass over one query-set version, as stored.

    ``corpus_items`` / ``corpus_chunks`` census the index the arm
    searched (items with >= 1 chunk / total chunks) at run time —
    results are only interpretable against the corpus that produced
    them, so the census is part of the record.
    """

    run_id: str
    set_version: str
    arm_key: str
    provider_model_id: str
    chunking: str
    status: str  # "completed" | "blocked"
    blocked_reason: str | None
    started_at: str
    corpus_items: int
    corpus_chunks: int


@dataclass(frozen=True, kw_only=True)
class CorpusItemRow:
    """The corpus_items fields the benchmark reads (read-only).

    A mirror of corpus_sync's row shape, kept local: the benchmark
    reads the production DB through a read-only connection and must
    not import a write-side repository to do it.
    """

    id: str
    source_kind: str
    source_ref: str
    student: str
    course: str | None
    title: str
    deep_link: str
    origin_url: str
    format: str
    content_hash: str | None
    fetched_at: str
    status: str
    error: str | None


@dataclass(frozen=True, kw_only=True)
class ArmSummary:
    """Derived metrics for one completed run (see benchmark.metrics).

    Hit/MRR figures are over scored queries (non-empty expected);
    empty-label queries are counted separately and summarized by
    the mean top score they drew (a graceful-behavior signal: low
    is honest, high means the arm confidently invents relevance).
    Separation is over queries where an expected AND an unexpected
    item both appear in the top-10 ranking.
    """

    arm_key: str
    queries_scored: int
    queries_empty_label: int
    resources_hit1: float
    resources_hit3: float
    resources_mrr: float
    passages_hit1: float
    passages_hit3: float
    passages_mrr: float
    top_chunk_hit1: float
    separation_resources: float | None
    separation_passages: float | None
    separation_resources_n: int
    separation_passages_n: int
    mean_latency_ms: float
    p50_latency_ms: float
    max_latency_ms: float
    empty_label_mean_top_score: float | None
