"""System test: the benchmark end to end over a fixture corpus.

Real pieces throughout — a pipeline-built corpus (EmbedStore +
chunker), the real CLI composition root, the real records store
and report — with one substitution at the external edge: the
embedding model (the deterministic fake provider stands in for
bge-small; Bedrock stands in as a credentials-less provider).
Retrieval quality is the live hal9k benchmark's job; this test
proves the machinery: arms run, records land, the report
re-derives the same numbers from the records.
"""

import json
from pathlib import Path

import pytest

from school_index.benchmark import cli
from school_index.benchmark.metrics import summarize
from school_index.benchmark.store import BenchmarkStore
from school_index.corpus_sync.lookup import Found
from tests.unit.benchmark.fakes import RaisingProvider
from tests.unit.benchmark.test_arms import _build_prod_corpus
from tests.unit.embed_index.fakes import FakeProvider


def test_benchmark_run_records_and_report_reproduces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, FakeProvider())
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            {
                "set_version": "sys1",
                "queries": [
                    {
                        "id": "q01-pyth",
                        "query": "pythagorean theorem",
                        "course": None,
                        "expected_item_ids": ["item-b"],
                        "notes": "fixture",
                    },
                    {
                        "id": "q02-absent",
                        "query": "water cycle",
                        "course": None,
                        "expected_item_ids": [],
                        "notes": "absent-topic probe",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    bench_root = tmp_path / "bench"
    monkeypatch.setenv("CORPUS_ROOT", str(prod))
    monkeypatch.setenv("BENCHMARK_ROOT", str(bench_root))
    monkeypatch.setenv("BENCHMARK_QUERIES", str(queries))
    monkeypatch.setattr(
        cli,
        "build_arm_providers",
        lambda *, models_dir: {
            "onnx-bge-small": FakeProvider(),
            "bedrock-titan": RaisingProvider(),
        },
    )

    assert cli.main(["run", "--arm", "all"]) == 0
    run_out = capsys.readouterr().out
    assert "onnx-production: resources hit@1" in run_out
    assert "bedrock-titan: BLOCKED" in run_out

    store = BenchmarkStore.open(bench_root)
    production = store.latest_run(arm_key="onnx-production")
    assert isinstance(production, Found)
    assert production.value.status == "completed"
    outcomes = store.outcomes(run_id=production.value.run_id)
    assert [o.query_id for o in outcomes] == ["q01-pyth", "q02-absent"]
    bedrock = store.latest_run(arm_key="bedrock-titan")
    assert isinstance(bedrock, Found)
    assert bedrock.value.status == "blocked"

    # The report re-derives the production summary from records
    # alone — the same hit@3 the run printed must appear in it.
    summary = summarize(arm_key="onnx-production", outcomes=outcomes)
    assert cli.main(["report"]) == 0
    report = capsys.readouterr().out
    assert f"{summary.resources_hit3:.2f}" in report
    assert "BLOCKED" in report
    assert production.value.run_id in report
