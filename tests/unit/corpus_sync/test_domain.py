"""Unit tests: corpus domain records + the Lookup sum type."""

import hashlib

import pytest

from school_index.corpus_sync.domain import ResourceRef, SyncReport
from school_index.corpus_sync.lookup import Found, Missing, MissingRecordError


def make_resource(**overrides: object) -> ResourceRef:
    fields: dict[str, object] = {
        "student_id": "sam",
        "student_name": "Sam",
        "course_name": "Algebra I",
        "title": "Unit 2 notes",
        "kind": "NOTES",
        "url": "https://docs.google.com/document/d/abc123/edit",
        "source_kind": "AGENDA",
        "scraped_at": "2026-10-23T05:15:00+00:00",
    }
    fields.update(overrides)
    return ResourceRef(**fields)  # type: ignore[arg-type]


def test_source_ref_is_student_plus_url() -> None:
    resource = make_resource()
    assert resource.source_ref == f"sam:{resource.url}"


def test_item_id_is_sha256_of_source_ref() -> None:
    resource = make_resource()
    expected = hashlib.sha256(resource.source_ref.encode("utf-8")).hexdigest()
    assert resource.item_id == expected
    # Deterministic across instances: the same resource re-harvested
    # resolves to the same store row.
    assert make_resource().item_id == resource.item_id


def test_item_id_separates_students_sharing_a_url() -> None:
    assert make_resource().item_id != make_resource(student_id="owen").item_id


def test_found_unwraps_its_value() -> None:
    assert Found(value=41).unwrap() == 41
    assert Found(value=41).found is True


def test_missing_unwrap_raises_with_reason() -> None:
    missing = Missing(reason="no such source_ref")
    assert missing.found is False
    with pytest.raises(MissingRecordError, match="no such source_ref"):
        missing.unwrap()


def test_failure_tally_counts_reasons() -> None:
    report = SyncReport(
        student="sam",
        total=3,
        downloaded=0,
        link_only=0,
        unchanged=0,
        failed=3,
        tombstoned=0,
        failures=(
            ("sam:https://a", "HTTP 403"),
            ("sam:https://b", "HTTP 403"),
            ("sam:https://c", "HTTP 404"),
        ),
    )
    assert report.failure_tally() == {"HTTP 403": 2, "HTTP 404": 1}
