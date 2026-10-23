"""Unit tests: the doc_extract CLI composition root.

The success path runs main() against a real on-disk corpus store
populated with one real PDF — the wiring, registry, runner, and
report printing are all real; there are no network collaborators
to fake in this package. The entry-point tests prove
``python -m school_index.doc_extract`` exits 2 on config failure.
"""

import hashlib
import os
import runpy
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from school_index.corpus_sync.domain import CorpusItem, ItemStatus
from school_index.corpus_sync.store import CorpusStore
from school_index.doc_extract import cli


def seed_pdf_item(root: Path, pdf_bytes: bytes) -> None:
    store = CorpusStore.open(root)
    source_ref = "sam:https://example.com/deck"
    item = CorpusItem(
        id=hashlib.sha256(source_ref.encode("utf-8")).hexdigest(),
        source_kind="AGENDA",
        source_ref=source_ref,
        student="sam",
        course="Algebra I",
        title="Deck",
        deep_link="https://example.com/deck",
        origin_url="https://example.com/deck",
        format="pdf",
        content_hash="hash-1",
        fetched_at="2026-10-23T13:00:00+00:00",
        status=ItemStatus.OK,
        error=None,
    )
    store.upsert_item(item)
    store.write_blob(item_id=item.id, content=pdf_bytes)


def test_main_extract_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    make_pdf: Callable[[list[str]], bytes],
) -> None:
    root = tmp_path / "corpus"
    seed_pdf_item(root, make_pdf(["Hello from the CLI test"]))
    monkeypatch.setenv("CORPUS_ROOT", str(root))
    assert cli.main(["extract"]) == 0
    out = capsys.readouterr().out
    assert "total=1" in out
    assert (
        "extracted=1 empty=0 failed=0 not_extractable=0 skipped=0 not_harvested=0"
        in out
    )


def test_main_extract_rerun_reports_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    make_pdf: Callable[[list[str]], bytes],
) -> None:
    root = tmp_path / "corpus"
    seed_pdf_item(root, make_pdf(["Hello again"]))
    monkeypatch.setenv("CORPUS_ROOT", str(root))
    assert cli.main(["extract"]) == 0
    capsys.readouterr()
    assert cli.main(["extract"]) == 0
    out = capsys.readouterr().out
    assert "extracted=0" in out
    assert "skipped=1" in out


def test_main_returns_2_without_corpus_root(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CORPUS_ROOT", raising=False)
    assert cli.main(["extract"]) == 2
    assert "CORPUS_ROOT" in capsys.readouterr().err


def test_main_guards_exit_with_mains_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORPUS_ROOT", raising=False)
    monkeypatch.setattr(sys, "argv", ["doc-extract", "extract"])
    cli_file = Path(str(cli.__file__))
    with pytest.raises(SystemExit) as package_exit:
        runpy.run_path(str(cli_file.parent / "__main__.py"), run_name="__main__")
    assert package_exit.value.code == 2
    with pytest.raises(SystemExit) as cli_exit:
        runpy.run_path(str(cli_file), run_name="__main__")
    assert cli_exit.value.code == 2


def test_module_entry_exits_2_without_config() -> None:
    env = {k: v for k, v in os.environ.items() if k != "CORPUS_ROOT"}
    completed = subprocess.run(
        [sys.executable, "-m", "school_index.doc_extract", "extract"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert completed.returncode == 2
    assert "CORPUS_ROOT" in completed.stderr
