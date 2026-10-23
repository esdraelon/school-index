"""System test: one full extraction pass, end to end.

Everything is real — the runner, the format registry, the pypdf
extractor, both repositories over an on-disk corpus store. There is
no network edge in this stage at all: extraction is local by
design, so nothing is stubbed. Assertions read the corpus.db with
raw SQL, independently of the repositories that wrote it.
"""

import hashlib
import io
import sqlite3
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from school_index.corpus_sync.domain import CorpusItem, ItemStatus
from school_index.corpus_sync.store import CorpusStore
from school_index.doc_extract.extractors import build_extractors
from school_index.doc_extract.runner import ExtractionRunner
from school_index.doc_extract.store import ExtractStore


def build_pdf(pages: list[str]) -> bytes:
    """A real PDF: one page per entry, "" entries text-free."""
    writer = PdfWriter()
    font = DictionaryObject()
    font.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    font_ref = writer._add_object(font)
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_ref
        resources = DictionaryObject()
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources
        if text:
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def seed_item(store: CorpusStore, url: str, **overrides: object) -> CorpusItem:
    source_ref = f"sam:{url}"
    fields: dict[str, object] = {
        "id": hashlib.sha256(source_ref.encode("utf-8")).hexdigest(),
        "source_kind": "AGENDA",
        "source_ref": source_ref,
        "student": "sam",
        "course": "Algebra I",
        "title": "Item",
        "deep_link": url,
        "origin_url": url,
        "format": "pdf",
        "content_hash": "hash-1",
        "fetched_at": "2026-10-23T13:00:00+00:00",
        "status": ItemStatus.OK,
        "error": None,
    }
    fields.update(overrides)
    item = CorpusItem(**fields)  # type: ignore[arg-type]
    store.upsert_item(item)
    return item


def test_full_extraction_and_noop_rerun(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    corpus = CorpusStore.open(root)
    good = seed_item(corpus, "https://example.com/deck")
    corpus.write_blob(
        item_id=good.id,
        content=build_pdf(["Systems of equations", "Graphing practice"]),
    )
    blank = seed_item(corpus, "https://example.com/scan")
    corpus.write_blob(item_id=blank.id, content=build_pdf([""]))
    corrupt = seed_item(corpus, "https://example.com/broken")
    corpus.write_blob(item_id=corrupt.id, content=b"%PDF-pretender, not really")
    seed_item(
        corpus,
        "https://example.com/ixl",
        format="external-link",
        content_hash=None,
    )
    seed_item(
        corpus,
        "https://example.com/walled",
        status=ItemStatus.FAILED,
        error="google sign-in page (file not publicly accessible)",
    )

    runner = ExtractionRunner(
        corpus_store=corpus,
        extract_store=ExtractStore.open(root),
        extractors=build_extractors(),
    )
    report = runner.run()
    assert (report.total, report.extracted, report.empty, report.failed) == (
        5,
        1,
        1,
        1,
    )
    assert (report.not_extractable, report.not_harvested, report.skipped) == (
        1,
        1,
        0,
    )

    # The database on disk says so, independently of the repositories.
    connection = sqlite3.connect(root / "corpus.db")
    census = dict(
        connection.execute(
            "SELECT outcome, COUNT(*) FROM extraction_state GROUP BY outcome"
        ).fetchall()
    )
    assert census == {
        "extracted": 1,
        "empty": 1,
        "failed": 1,
        "not-extractable": 1,
    }
    text = connection.execute(
        "SELECT text FROM item_text WHERE item_id = ?", (good.id,)
    ).fetchone()[0]
    assert text == (
        "[[page 1]]\nSystems of equations\n[[page 2]]\nGraphing practice"
    )
    empty_text = connection.execute(
        "SELECT text FROM item_text WHERE item_id = ?", (blank.id,)
    ).fetchone()[0]
    assert empty_text == ""
    failed_error = connection.execute(
        "SELECT error FROM extraction_state WHERE item_id = ?", (corrupt.id,)
    ).fetchone()[0]
    assert failed_error  # a real error record, not a silent skip
    health = connection.execute(
        "SELECT last_status FROM sync_state WHERE source = 'doc-extract'"
    ).fetchone()
    assert health == ("partial",)
    stamps = dict(
        connection.execute(
            "SELECT item_id, extracted_at FROM extraction_state"
        ).fetchall()
    )

    rerun = runner.run()
    assert (rerun.extracted, rerun.empty, rerun.not_extractable) == (0, 0, 0)
    assert rerun.skipped == 3
    assert rerun.failed == 1  # the corrupt file retries, by design
    # Skipped items keep their first-run stamps (the failure row
    # legitimately re-stamps — it retried).
    good_stamp = connection.execute(
        "SELECT extracted_at FROM extraction_state WHERE item_id = ?",
        (good.id,),
    ).fetchone()[0]
    assert good_stamp == stamps[good.id]
