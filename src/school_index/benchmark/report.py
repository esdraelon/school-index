"""The benchmark report: summaries re-derived from records alone.

Pattern: pure rendering over the records store. Given only
benchmark.db, this reproduces every number a run printed — the M5
gate ("benchmark reproducible from the stored records") is this
module's existence proof. Numbers come from benchmark.metrics over
stored outcomes; nothing is recomputed from the corpus.

The failure listing shows the misses a tutor would actually feel:
scored queries where no expected item reached the resources top-3,
with the top-3 the arm DID return (8-char item-id prefixes —
titles live in the corpus census, and the milestone writeup maps
the interesting ones).
"""

from school_index.benchmark.domain import RunRecord
from school_index.benchmark.metrics import summarize
from school_index.benchmark.store import BenchmarkStore

_SHORT = 8


def _short(item_id: str) -> str:
    return item_id[:_SHORT]


def _sep(value: float | None) -> str:
    return f"{value:+.3f}" if value is not None else "—"


def _latest_runs(store: BenchmarkStore) -> list[RunRecord]:
    """The most recent run per arm, in arm-key order."""
    latest: dict[str, RunRecord] = {}
    for run in store.runs():  # oldest first; later rows overwrite
        latest[run.arm_key] = run
    return [latest[key] for key in sorted(latest)]


def render_report(store: BenchmarkStore) -> str:
    """The full markdown report over the latest run of each arm."""
    runs = _latest_runs(store)
    if not runs:
        return "No benchmark runs recorded.\n"
    lines = ["# Benchmark report (derived from benchmark.db records)", ""]
    lines.append(
        "| Arm | Status | Model | Res hit@1 | Res hit@3 | Res MRR | "
        "Pas hit@1 | Pas hit@3 | Pas MRR | Top-chunk | Sep res | Sep pas | "
        "Mean ms | Empty-top |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    completed: list[RunRecord] = []
    for run in runs:
        if run.status == "blocked":
            lines.append(
                f"| {run.arm_key} | BLOCKED | {run.provider_model_id} | "
                + " | ".join(["—"] * 11)
                + " |"
            )
            continue
        completed.append(run)
        summary = summarize(
            arm_key=run.arm_key, outcomes=store.outcomes(run_id=run.run_id)
        )
        empty_top = (
            f"{summary.empty_label_mean_top_score:.3f}"
            if summary.empty_label_mean_top_score is not None
            else "—"
        )
        lines.append(
            f"| {run.arm_key} | completed | {run.provider_model_id} "
            f"| {summary.resources_hit1:.2f} | {summary.resources_hit3:.2f} "
            f"| {summary.resources_mrr:.3f} "
            f"| {summary.passages_hit1:.2f} | {summary.passages_hit3:.2f} "
            f"| {summary.passages_mrr:.3f} "
            f"| {summary.top_chunk_hit1:.2f} "
            f"| {_sep(summary.separation_resources)} "
            f"| {_sep(summary.separation_passages)} "
            f"| {summary.mean_latency_ms:.0f} | {empty_top} |"
        )
    for run in runs:
        lines.append("")
        lines.append(
            f"## {run.arm_key} — run {run.run_id} "
            f"(set v{run.set_version}, {run.started_at})"
        )
        lines.append(
            f"Corpus: {run.corpus_items} items / {run.corpus_chunks} chunks. "
            f"Chunking: {run.chunking or '—'}."
        )
        if run.status == "blocked":
            lines.append(f"BLOCKED: {run.blocked_reason}")
            continue
        outcomes = store.outcomes(run_id=run.run_id)
        misses = [
            outcome
            for outcome in outcomes
            if outcome.expected_item_ids
            and not any(
                r.item_id in set(outcome.expected_item_ids)
                for r in outcome.resources[:3]
            )
        ]
        if not misses:
            lines.append("No resources hit@3 misses.")
        else:
            lines.append(f"Resources hit@3 misses ({len(misses)}):")
            for outcome in misses:
                expected = ", ".join(_short(i) for i in outcome.expected_item_ids)
                returned = ", ".join(
                    _short(r.item_id) for r in outcome.resources[:3]
                )
                lines.append(
                    f"- {outcome.query_id}: expected [{expected}] "
                    f"— returned top-3 [{returned}]"
                )
    return "\n".join(lines) + "\n"
