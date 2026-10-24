"""Unit tests: the CLI composition root.

The success path runs main() with the two network collaborators
(MCP source, HTTP downloader) swapped for fakes at the module seam —
the wiring, store, registry, and report printing are real. The
entry-point test runs the module as a subprocess to prove
``python -m school_index.corpus_sync`` exits 2 on config failure.
"""

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from school_index.corpus_sync import cli
from school_index.corpus_sync.client import PayloadError
from school_index.corpus_sync.domain import ListedResources, ResourceRef
from school_index.corpus_sync.fetch import RawResponse

DECK_URL = "https://docs.google.com/presentation/d/deck1/pub"
DECK_EXPORT = "https://docs.google.com/presentation/d/deck1/export?format=pdf"
IXL_URL = "https://www.ixl.com/math/algebra-1/pythagorean-theorem"
WALLED_URL = "https://docs.google.com/document/d/doc1/edit"
WALLED_EXPORT = "https://docs.google.com/document/d/doc1/export?format=pdf"


def fixture_resources() -> list[ResourceRef]:
    base = {
        "student_id": "sam",
        "student_name": "Sam",
        "course_name": "Algebra I",
        "source_kind": "AGENDA",
        "scraped_at": "2026-10-23T05:15:00+00:00",
    }
    return [
        ResourceRef(title="Deck", kind="DOCUMENT", url=DECK_URL, **base),  # type: ignore[arg-type]
        ResourceRef(title="IXL", kind="TUTORIAL", url=IXL_URL, **base),  # type: ignore[arg-type]
        ResourceRef(title="Walled", kind="DOCUMENT", url=WALLED_URL, **base),  # type: ignore[arg-type]
    ]


class FakeSource:
    def __init__(self, *, url: str, token: str) -> None:
        pass

    def list_resources(self, *, student: str) -> ListedResources:
        resources = fixture_resources()
        return ListedResources(
            resources=tuple(resources),
            probe_distinct=len(resources),
            partition_calls=1,
            exhaustive=True,
        )


class ExplodingSource:
    def __init__(self, *, url: str, token: str) -> None:
        pass

    def list_resources(self, *, student: str) -> ListedResources:
        raise PayloadError("contract broke")


class FakeDownloader:
    def get(self, *, url: str) -> RawResponse:
        if url == WALLED_EXPORT:
            return RawResponse(status_code=403, headers={}, content=b"")
        assert url == DECK_EXPORT
        return RawResponse(
            status_code=200,
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF-deck",
        )


def test_main_sync_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SCHOOL_MCP_TOKEN", "test-token")
    monkeypatch.setenv("CORPUS_ROOT", str(tmp_path / "corpus"))
    monkeypatch.setattr(cli, "McpResourceSource", FakeSource)
    monkeypatch.setattr(cli, "HttpDownloader", FakeDownloader)
    assert cli.main(["sync", "--student", "sam"]) == 0
    out = capsys.readouterr().out
    assert "total=3" in out
    assert "coverage exhaustive=True probe_distinct=3 partition_calls=1" in out
    assert "downloaded=1 link_only=1 unchanged=0 failed=1 tombstoned=0" in out
    assert "failure: HTTP 403 ×1" in out
    assert (tmp_path / "corpus" / "corpus.db").exists()


def test_main_returns_2_on_contract_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SCHOOL_MCP_TOKEN", "test-token")
    monkeypatch.setenv("CORPUS_ROOT", str(tmp_path / "corpus"))
    monkeypatch.setattr(cli, "McpResourceSource", ExplodingSource)
    monkeypatch.setattr(cli, "HttpDownloader", FakeDownloader)
    assert cli.main(["sync", "--student", "sam"]) == 2
    assert "contract error" in capsys.readouterr().err


def test_main_guards_exit_with_mains_code(monkeypatch: pytest.MonkeyPatch) -> None:
    # Execute both __main__ shims in-process, with __name__ set the way
    # the interpreter sets it, so the guard lines run as real code:
    # config failure makes main() return 2 and the guard raises it.
    monkeypatch.delenv("SCHOOL_MCP_TOKEN", raising=False)
    monkeypatch.delenv("CORPUS_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", ["corpus-sync", "sync"])
    cli_file = Path(str(cli.__file__))
    with pytest.raises(SystemExit) as package_exit:
        runpy.run_path(str(cli_file.parent / "__main__.py"), run_name="__main__")
    assert package_exit.value.code == 2
    with pytest.raises(SystemExit) as cli_exit:
        runpy.run_path(str(cli_file), run_name="__main__")
    assert cli_exit.value.code == 2


def test_module_entry_exits_2_without_config(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SCHOOL_MCP")}
    env.pop("CORPUS_ROOT", None)
    completed = subprocess.run(
        [sys.executable, "-m", "school_index.corpus_sync", "sync"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert completed.returncode == 2
    assert "SCHOOL_MCP_TOKEN" in completed.stderr
