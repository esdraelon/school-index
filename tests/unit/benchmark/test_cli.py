"""Unit tests: the benchmark CLI composition root.

Real config/store/runner throughout; the model providers are the
only substitution (patched at the composition root, the house
pattern) because the real ONNX model is an external artifact.
"""

import json
from pathlib import Path

import pytest

from school_index.benchmark import cli
from school_index.benchmark.domain import ArmSummary
from school_index.benchmark.store import BenchmarkStore
from school_index.corpus_sync.lookup import Found
from tests.unit.benchmark.fakes import RaisingProvider
from tests.unit.benchmark.test_arms import _build_prod_corpus
from tests.unit.embed_index.fakes import FakeProvider


def _summary(hit3: float) -> ArmSummary:
    return ArmSummary(
        arm_key="onnx-production",
        queries_scored=29,
        queries_empty_label=4,
        resources_hit1=0.0,
        resources_hit3=hit3,
        resources_mrr=0.0,
        passages_hit1=0.0,
        passages_hit3=0.0,
        passages_mrr=0.0,
        top_chunk_hit1=0.0,
        separation_resources=None,
        separation_passages=None,
        separation_resources_n=0,
        separation_passages_n=0,
        mean_latency_ms=0.0,
        p50_latency_ms=0.0,
        max_latency_ms=0.0,
        empty_label_mean_top_score=None,
    )


@pytest.mark.parametrize(
    "summary,expected",
    [(None, False), (_summary(0.69), True), (_summary(0.70), False), (_summary(0.97), False)],
)
def test_chunk600_trigger_threshold(
    summary: ArmSummary | None, expected: bool
) -> None:
    assert cli.chunk600_triggered(summary) is expected


def _env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corpus_root: Path
) -> Path:
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            {
                "set_version": "t1",
                "queries": [
                    {
                        "id": "q01-a",
                        "query": "pythagorean theorem",
                        "course": None,
                        "expected_item_ids": ["item-b"],
                        "notes": "fixture",
                    },
                    {
                        "id": "q02-b",
                        "query": "slope intercept",
                        "course": None,
                        "expected_item_ids": ["item-a"],
                        "notes": "fixture",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CORPUS_ROOT", str(corpus_root))
    monkeypatch.setenv("BENCHMARK_ROOT", str(tmp_path / "bench"))
    monkeypatch.setenv("BENCHMARK_QUERIES", str(queries))
    return tmp_path / "bench"


def _patch_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "build_arm_providers",
        lambda *, models_dir: {
            "onnx-bge-small": FakeProvider(),
            "bedrock-titan": RaisingProvider(),
        },
    )


def test_report_on_empty_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _env(monkeypatch, tmp_path, tmp_path / "corpus")
    assert cli.main(["report"]) == 0
    assert "No benchmark runs recorded." in capsys.readouterr().out


def test_config_error_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CORPUS_ROOT", raising=False)
    assert cli.main(["report"]) == 2
    assert "config error" in capsys.readouterr().err


def test_missing_query_set_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _env(monkeypatch, tmp_path, tmp_path / "corpus")
    monkeypatch.setenv("BENCHMARK_QUERIES", str(tmp_path / "absent.json"))
    assert cli.main(["run", "--arm", "onnx-production"]) == 2
    assert "query set error" in capsys.readouterr().err


def test_missing_corpus_db_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _env(monkeypatch, tmp_path, tmp_path / "no-corpus-here")
    _patch_providers(monkeypatch)
    assert cli.main(["run", "--arm", "onnx-production"]) == 2
    assert "corpus error" in capsys.readouterr().err


def test_run_production_arm_records_a_completed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, FakeProvider())
    bench_root = _env(monkeypatch, tmp_path, prod)
    _patch_providers(monkeypatch)
    assert cli.main(["run", "--arm", "onnx-production"]) == 0
    out = capsys.readouterr().out
    assert "onnx-production: resources hit@1" in out
    store = BenchmarkStore.open(bench_root)
    latest = store.latest_run(arm_key="onnx-production")
    assert isinstance(latest, Found)
    assert latest.value.status == "completed"
    assert len(store.outcomes(run_id=latest.value.run_id)) == 2


def test_malformed_query_set_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _env(monkeypatch, tmp_path, tmp_path / "corpus")
    bad = tmp_path / "bad.json"
    bad.write_text('{"set_version": 3, "queries": []}', encoding="utf-8")
    monkeypatch.setenv("BENCHMARK_QUERIES", str(bad))
    assert cli.main(["run", "--arm", "onnx-production"]) == 2
    assert "query set error" in capsys.readouterr().err


def test_index_not_ready_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, FakeProvider())
    _env(monkeypatch, tmp_path, prod)
    _patch_providers(monkeypatch)

    def not_ready(*args: object, **kwargs: object) -> object:
        raise cli.IndexNotReady("no corpus.db under fixture")

    monkeypatch.setattr(cli, "build_arm", not_ready)
    assert cli.main(["run", "--arm", "onnx-production"]) == 2
    assert "index error" in capsys.readouterr().err


def test_run_all_with_production_blocked_skips_chunking_as_na(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, FakeProvider())
    bench_root = _env(monkeypatch, tmp_path, prod)
    monkeypatch.setattr(
        cli,
        "build_arm_providers",
        lambda *, models_dir: {
            "onnx-bge-small": RaisingProvider(),
            "bedrock-titan": RaisingProvider(),
        },
    )
    assert cli.main(["run", "--arm", "all"]) == 0
    out = capsys.readouterr().out
    assert "onnx-production: BLOCKED" in out
    assert "chunking arm skipped" in out and "n/a" in out
    store = BenchmarkStore.open(bench_root)
    production = store.latest_run(arm_key="onnx-production")
    assert isinstance(production, Found)
    assert production.value.status == "blocked"


def test_run_all_triggers_chunking_arm_below_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, FakeProvider())
    bench_root = _env(monkeypatch, tmp_path, prod)
    _patch_providers(monkeypatch)
    # The decision RULE is pinned by test_chunk600_trigger_threshold;
    # here the summary is stubbed at the composition root so the
    # wiring (trigger line + the real variant arm actually running)
    # is what's under test.
    monkeypatch.setattr(cli, "_production_summary", lambda store: _summary(0.5))
    assert cli.main(["run", "--arm", "all"]) == 0
    out = capsys.readouterr().out
    assert "chunking arm triggered" in out
    store = BenchmarkStore.open(bench_root)
    variant = store.latest_run(arm_key="onnx-chunk600")
    assert isinstance(variant, Found)
    assert variant.value.status == "completed"
    assert "target_chars=600" in variant.value.chunking


def test_run_all_records_bedrock_blocked_and_decides_chunking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prod = tmp_path / "prod"
    _build_prod_corpus(prod, FakeProvider())
    bench_root = _env(monkeypatch, tmp_path, prod)
    _patch_providers(monkeypatch)
    assert cli.main(["run", "--arm", "all"]) == 0
    out = capsys.readouterr().out
    assert "bedrock-titan: BLOCKED" in out
    assert "Unable to locate credentials" in out
    # The chunking decision is printed either way (its direction
    # depends on the fake provider's rankings; the decision RULE is
    # pinned by test_chunk600_trigger_threshold).
    assert "chunking arm" in out
    store = BenchmarkStore.open(bench_root)
    blocked = store.latest_run(arm_key="bedrock-titan")
    assert isinstance(blocked, Found)
    assert blocked.value.status == "blocked"
