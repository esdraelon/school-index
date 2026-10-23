"""Unit tests: the index_mcp composition root.

The provider registry is swapped for the stub and uvicorn.run is
recorded instead of executed — the composition root is the one
place allowed to choose collaborators, so it is the seam the tests
use (the embed CLI tests' pattern). Everything else is real: the
config builder, the read store, the census check.
"""

from pathlib import Path
from typing import Any

import pytest

from school_index.corpus_sync.store import CorpusStore
from school_index.index_mcp import cli
from tests.unit.index_mcp.fakes import DECK_A, SeedItem, StubProvider, build_index


class UvicornRecorder:
    """Records uvicorn.run calls instead of serving."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, app: object, **kwargs: Any) -> None:
        self.calls.append({"app": app, **kwargs})


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> UvicornRecorder:
    recorder = UvicornRecorder()
    monkeypatch.setattr(cli.uvicorn, "run", recorder)
    monkeypatch.setattr(
        cli,
        "build_providers",
        lambda *, models_dir: {"onnx-bge-small": StubProvider()},
    )
    return recorder


def _env(monkeypatch: pytest.MonkeyPatch, root: Path, **extra: str) -> None:
    monkeypatch.setenv("INDEX_MCP_TOKEN", "test-token")
    monkeypatch.setenv("CORPUS_ROOT", str(root))
    for key, value in extra.items():
        monkeypatch.setenv(key, value)


def test_serves_with_proxy_header_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: UvicornRecorder
) -> None:
    build_index(tmp_path, [DECK_A])
    _env(monkeypatch, tmp_path)
    assert cli.main([]) == 0
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 8766
    assert call["proxy_headers"] is True
    assert call["forwarded_allow_ips"] == "*"


def test_missing_token_is_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: UvicornRecorder
) -> None:
    monkeypatch.setenv("CORPUS_ROOT", str(tmp_path))
    monkeypatch.delenv("INDEX_MCP_TOKEN", raising=False)
    assert cli.main([]) == 2
    assert recorder.calls == []


def test_missing_corpus_db_is_index_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: UvicornRecorder
) -> None:
    _env(monkeypatch, tmp_path)
    assert cli.main([]) == 2
    assert recorder.calls == []


def test_unknown_model_key_is_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: UvicornRecorder
) -> None:
    build_index(tmp_path, [DECK_A])
    _env(monkeypatch, tmp_path, EMBED_MODEL="nope")
    assert cli.main([]) == 2
    assert recorder.calls == []


def test_model_census_mismatch_refuses_to_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: UvicornRecorder
) -> None:
    foreign = SeedItem(
        url="https://example.com/foreign",
        title="Foreign",
        course=None,
        model_id="another-model",
        chunks=DECK_A.chunks,
    )
    build_index(tmp_path, [foreign])
    _env(monkeypatch, tmp_path)
    assert cli.main([]) == 2
    assert recorder.calls == []


def test_empty_index_warns_and_serves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorder: UvicornRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    CorpusStore.open(tmp_path)  # schema, nothing embedded
    _env(monkeypatch, tmp_path)
    assert cli.main([]) == 0
    assert len(recorder.calls) == 1
    assert "no embedded items" in capsys.readouterr().err
