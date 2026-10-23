"""Shared fixtures for doc_extract unit tests: real PDF bytes.

Fixture PDFs are generated with pypdf's own writer (a blank page
per entry, plus a text content stream for non-empty entries), so
the extractor under test reads exactly what a producer would write
— no checked-in binary blobs and no mocking of the PDF layer.
"""

import io
from collections.abc import Callable

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def build_pdf(pages: list[str]) -> bytes:
    """A PDF with one page per entry; "" entries are text-free pages."""
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


@pytest.fixture
def make_pdf() -> Callable[[list[str]], bytes]:
    return build_pdf
