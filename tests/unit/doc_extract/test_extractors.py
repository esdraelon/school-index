"""Unit tests: the extractor registry + the pypdf strategy."""

from collections.abc import Callable
from pathlib import Path

from school_index.doc_extract.extractors import (
    ExtractedText,
    ExtractionFailed,
    PdfExtractor,
    build_extractors,
    render_pages,
)


def test_registry_covers_exactly_pdf_in_v1() -> None:
    registry = build_extractors()
    assert set(registry) == {"pdf"}
    assert isinstance(registry["pdf"], PdfExtractor)
    assert registry["pdf"].name == "pypdf"


def test_registry_miss_for_future_formats_is_a_lookup_miss() -> None:
    # html/docx/native slides slot in later by adding entries; until
    # then the runner resolves them to NOT_EXTRACTABLE via .get().
    registry = build_extractors()
    assert registry.get("html") is None
    assert registry.get("docx") is None
    assert registry.get("external-link") is None


def test_pdf_pages_come_back_in_reading_order(
    tmp_path: Path, make_pdf: Callable[[list[str]], bytes]
) -> None:
    path = tmp_path / "deck.pdf"
    path.write_bytes(
        make_pdf(["Pythagorean theorem page one", "Worked example page two", ""])
    )
    outcome = PdfExtractor().extract(path=path)
    assert isinstance(outcome, ExtractedText)
    assert outcome.pages == (
        "Pythagorean theorem page one",
        "Worked example page two",
        "",
    )
    assert outcome.page_count == 3
    assert not outcome.is_empty


def test_render_pages_marks_every_page_boundary(
    tmp_path: Path, make_pdf: Callable[[list[str]], bytes]
) -> None:
    path = tmp_path / "deck.pdf"
    path.write_bytes(make_pdf(["Alpha", "Beta", ""]))
    outcome = PdfExtractor().extract(path=path)
    assert isinstance(outcome, ExtractedText)
    assert render_pages(outcome.pages) == (
        "[[page 1]]\nAlpha\n[[page 2]]\nBeta\n[[page 3]]\n"
    )
    assert render_pages([]) == ""
    assert render_pages(["only"]) == "[[page 1]]\nonly"


def test_blank_pdf_is_extracted_but_empty(
    tmp_path: Path, make_pdf: Callable[[list[str]], bytes]
) -> None:
    path = tmp_path / "image-only.pdf"
    path.write_bytes(make_pdf(["", ""]))
    outcome = PdfExtractor().extract(path=path)
    assert isinstance(outcome, ExtractedText)
    assert outcome.is_empty
    assert outcome.page_count == 2


def test_corrupt_pdf_fails_as_data_not_exception(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf at all, despite the name")
    outcome = PdfExtractor().extract(path=path)
    assert isinstance(outcome, ExtractionFailed)
    assert ":" in outcome.error
    assert "\n" not in outcome.error


def test_missing_file_fails_as_data_not_exception(tmp_path: Path) -> None:
    outcome = PdfExtractor().extract(path=tmp_path / "gone.pdf")
    assert isinstance(outcome, ExtractionFailed)
    assert outcome.error.startswith("FileNotFoundError")
