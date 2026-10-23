"""Unit tests: the embed CLI composition root.

The provider registry is swapped for the deterministic fake (the
composition root is the one place allowed to choose collaborators,
so it is the seam the tests use); everything else — stores, runner,
chunker — is real over a temp corpus root.
"""

import hashlib
from pathlib import Path

import pytest

from school_index.corpus_sync.domain import CorpusItem, ItemStatus
from school_index.corpus_sync.store import CorpusStore
from school_index.doc_extract.domain import ExtractionOutcome, ExtractionState
from school_index.doc_extract.store import ExtractStore
from school_index.embed_index import cli
from tests.unit.embed_index.fakes import FakeProvider


def seed_root(root: Path) -> None:
    corpus = CorpusStore.open(root)
    source_ref = "sam:https://example.com/deck"
    item = CorpusItem(
        id=hashlib.sha256(source_ref.encode("utf-8")).hexdigest(),
        source_kind="AGENDA",
        source_ref=source_ref,
        student="sam",
        course="Algebra I",
        title="Systems Deck",
        deep_link="https://example.com/deck",
        origin_url="https://example.com/deck",
        format="pdf",
        content_hash="hash-1",
        fetched_at="2026-10-23T13:00:00+00:00",
        status=ItemStatus.OK,
        error=None,
    )
    corpus.upsert_item(item)
    ExtractStore.open(root).record_outcome(
        ExtractionState(
            item_id=item.id,
            outcome=ExtractionOutcome.EXTRACTED,
            extractor="pypdf",
            source_hash="hash-1",
            format="pdf",
            char_count=30,
            page_count=1,
            error=None,
            extracted_at="2026-10-23T14:00:00+00:00",
        ),
        text="[[page 1]]\nSystems of equations and graphing.",
    )


@pytest.fixture
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "corpus"
    seed_root(root)
    monkeypatch.setenv("CORPUS_ROOT", str(root))
    monkeypatch.setattr(
        cli,
        "build_providers",
        lambda *, models_dir: {"onnx-bge-small": FakeProvider()},
    )
    return root


def test_build_then_search(
    wired: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["build"]) == 0
    out = capsys.readouterr().out
    assert "embedded=1" in out
    assert "chunks_total=1" in out
    assert "backend=" in out

    assert cli.main(["build"]) == 0
    out = capsys.readouterr().out
    assert "embedded=0" in out
    assert "skipped=1" in out

    assert cli.main(["search", "systems of equations"]) == 0
    out = capsys.readouterr().out
    assert "Systems Deck" in out
    assert "page 1" in out


def test_missing_config_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORPUS_ROOT", raising=False)
    assert cli.main(["build"]) == 2


def test_unknown_model_exits_2(
    wired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EMBED_MODEL", "no-such-model")
    assert cli.main(["build"]) == 2


class MissingModelProvider(FakeProvider):
    def embed_query(self, text):  # type: ignore[override]
        from school_index.embed_index.providers import ModelNotDownloaded

        raise ModelNotDownloaded("not here")


class ExplodingProvider(FakeProvider):
    def embed_documents(self, texts):  # type: ignore[override]
        raise RuntimeError("boom")


class OtherModelProvider(FakeProvider):
    model_id = "other-model"


def _swap_provider(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> None:
    monkeypatch.setattr(
        cli,
        "build_providers",
        lambda *, models_dir: {"onnx-bge-small": provider},
    )


def test_search_without_model_exits_2(
    wired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _swap_provider(monkeypatch, MissingModelProvider())
    assert cli.main(["search", "anything"]) == 2


def test_search_empty_index_reports_no_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "empty"
    CorpusStore.open(root)
    monkeypatch.setenv("CORPUS_ROOT", str(root))
    _swap_provider(monkeypatch, FakeProvider())
    assert cli.main(["search", "anything"]) == 0
    assert "no hits" in capsys.readouterr().out


def test_build_with_failing_provider_reports_failures(
    wired: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _swap_provider(monkeypatch, ExplodingProvider())
    assert cli.main(["build"]) == 0
    out = capsys.readouterr().out
    assert "failed=1" in out
    assert "failure: RuntimeError: boom ×1" in out


def test_build_model_mismatch_exits_2(
    wired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.main(["build"]) == 0  # states now name fake-hash
    _swap_provider(monkeypatch, OtherModelProvider())
    assert cli.main(["build"]) == 2
    assert cli.main(["build", "--rebuild"]) == 0


def test_fetch_model_delegates(
    wired: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Path] = []

    def fake_download(models_dir: Path) -> Path:
        seen.append(models_dir)
        return models_dir / "bge-small-en-v1.5"

    monkeypatch.setattr(cli, "download_model", fake_download)
    assert cli.main(["fetch-model"]) == 0
    assert seen == [wired / "models"]
    assert "model ready" in capsys.readouterr().out
