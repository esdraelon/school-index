"""CLI composition root for the benchmark: run arms, print reports.

Pattern: Composition Root — the only benchmark module that reads
``os.environ`` and wires real collaborators (arm providers, the
records store, the runner).

Usage:
  python -m school_index.benchmark run --arm onnx-production
  python -m school_index.benchmark run --arm all
  python -m school_index.benchmark report

``--arm all`` runs production, then Bedrock (recorded blocked when
it cannot run — a result, not a failure), then decides the
chunking arm from arm A's recorded numbers: the variant runs only
when production's resources hit@3 falls below the trigger (pointer
accuracy that looks chunk-bound), per the M5 checklist. Any arm
can always be run explicitly, trigger or not.
"""

import argparse
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone

from school_index.benchmark import corpus_read
from school_index.benchmark.arms import (
    ARM_SPECS,
    ArmBlocked,
    build_arm,
    build_arm_providers,
)
from school_index.benchmark.config import BenchmarkConfig, ConfigError, config_from_env
from school_index.benchmark.domain import ArmSummary, QuerySet
from school_index.benchmark.metrics import summarize
from school_index.benchmark.queryset import QuerySetError, load_query_set
from school_index.benchmark.report import render_report
from school_index.benchmark.runner import BenchmarkRunner
from school_index.benchmark.store import BenchmarkStore
from school_index.corpus_sync.lookup import Found
from school_index.embed_index.providers import EmbeddingProvider
from school_index.index_mcp.store import IndexNotReady

#: Arm C's trigger: production resources hit@3 below this is what
#: "chunk-bound" looks like (M5 checklist: "say, hit@3 < 0.7").
CHUNK600_TRIGGER_HIT3 = 0.7


def chunk600_triggered(summary: ArmSummary | None) -> bool:
    """True when arm A's pointer accuracy justifies the chunking arm."""
    return summary is not None and summary.resources_hit3 < CHUNK600_TRIGGER_HIT3


def _run_one(
    arm_key: str,
    *,
    config: BenchmarkConfig,
    query_set: QuerySet,
    runner: BenchmarkRunner,
    store: BenchmarkStore,
    providers: Mapping[str, EmbeddingProvider],
    census: tuple[int, int],
) -> str:
    """Run (or record-blocked) one arm; return its one-line outcome."""
    spec = ARM_SPECS[arm_key]
    moment = datetime.now(timezone.utc)
    started_at = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = moment.strftime("%Y%m%dT%H%M%S") + f"{moment.microsecond:06d}"
    run_id = f"{arm_key}-v{query_set.set_version}-{stamp}"
    corpus_items, corpus_chunks = census
    try:
        arm = build_arm(
            spec, config=config, providers=providers, now=lambda: started_at
        )
    except ArmBlocked as blocked:
        provider = providers.get(spec.provider_key)
        runner.record_blocked(
            spec=spec,
            provider_model_id=(
                provider.model_id if provider is not None else spec.provider_key
            ),
            query_set=query_set,
            corpus_items=corpus_items,
            corpus_chunks=corpus_chunks,
            reason=blocked.reason,
            run_id=run_id,
            started_at=started_at,
        )
        return f"{arm_key}: BLOCKED — {blocked.reason}"
    record = runner.run_arm(
        arm=arm,
        query_set=query_set,
        corpus_items=corpus_items,
        corpus_chunks=corpus_chunks,
        run_id=run_id,
        started_at=started_at,
    )
    summary = summarize(
        arm_key=arm_key, outcomes=store.outcomes(run_id=record.run_id)
    )
    return (
        f"{arm_key}: resources hit@1 {summary.resources_hit1:.2f} "
        f"hit@3 {summary.resources_hit3:.2f} MRR {summary.resources_mrr:.3f}; "
        f"passages hit@3 {summary.passages_hit3:.2f}; "
        f"mean latency {summary.mean_latency_ms:.0f} ms (run {record.run_id})"
    )


def _production_summary(store: BenchmarkStore) -> ArmSummary | None:
    """Arm A's latest completed summary, re-derived from records."""
    latest = store.latest_run(arm_key="onnx-production")
    if not isinstance(latest, Found) or latest.value.status != "completed":
        return None
    return summarize(
        arm_key="onnx-production",
        outcomes=store.outcomes(run_id=latest.value.run_id),
    )


def main(argv: list[str] | None = None) -> int:
    """Wire the real collaborators and execute the subcommand."""
    parser = argparse.ArgumentParser(prog="benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run one arm (or all, in order)")
    run_parser.add_argument(
        "--arm", required=True, choices=[*sorted(ARM_SPECS), "all"]
    )
    sub.add_parser("report", help="re-derive the report from stored records")
    args = parser.parse_args(argv)

    try:
        config = config_from_env(os.environ)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    store = BenchmarkStore.open(config.benchmark_root)

    if args.command == "report":
        print(render_report(store), end="")
        return 0

    try:
        loaded = load_query_set(config.queries_path)
    except QuerySetError as exc:
        print(f"query set error: {exc}", file=sys.stderr)
        return 2
    if not isinstance(loaded, Found):
        print(f"query set error: {loaded.reason}", file=sys.stderr)
        return 2
    query_set = loaded.value

    try:
        census = corpus_read.census(config.corpus_root)
    except Exception as exc:  # a missing/unreadable corpus DB is a setup error
        print(
            f"corpus error: cannot census {config.corpus_root}: {exc}",
            file=sys.stderr,
        )
        return 2
    providers = build_arm_providers(models_dir=config.models_dir)
    runner = BenchmarkRunner(store=store)

    arm_keys = (
        ["onnx-production", "bedrock-titan"] if args.arm == "all" else [args.arm]
    )
    try:
        for arm_key in arm_keys:
            print(
                _run_one(
                    arm_key,
                    config=config,
                    query_set=query_set,
                    runner=runner,
                    store=store,
                    providers=providers,
                    census=census,
                )
            )
        if args.arm == "all":
            summary = _production_summary(store)
            if chunk600_triggered(summary):
                assert summary is not None
                print(
                    f"chunking arm triggered: production resources hit@3 "
                    f"{summary.resources_hit3:.2f} < {CHUNK600_TRIGGER_HIT3}"
                )
                print(
                    _run_one(
                        "onnx-chunk600",
                        config=config,
                        query_set=query_set,
                        runner=runner,
                        store=store,
                        providers=providers,
                        census=census,
                    )
                )
            else:
                seen = f"{summary.resources_hit3:.2f}" if summary else "n/a"
                print(
                    f"chunking arm skipped: production resources hit@3 {seen} "
                    f">= {CHUNK600_TRIGGER_HIT3} (not chunk-bound)"
                )
    except IndexNotReady as exc:
        print(f"index error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
