"""The labeled query set: loading + validation of queries.json.

Pattern: Adapter (file → domain) with Lookup at the boundary — a
missing file is ``Missing`` (the CLI reports it plainly); a present
but malformed file raises ``QuerySetError``, because silently
scoring against a half-valid set would manufacture evidence.
Validation is strict about the things labels depend on: a set
version, unique query ids, non-empty query text, and expected ids
as plain strings (membership is checked against the live corpus
at run time, not here — this module knows nothing of the corpus).
"""

import json
from pathlib import Path

from school_index.benchmark.domain import QueryCase, QuerySet
from school_index.corpus_sync.lookup import Found, Lookup, Missing


class QuerySetError(ValueError):
    """queries.json exists but is not a valid labeled set."""


def _parse_case(raw: object, *, index: int) -> QueryCase:
    if not isinstance(raw, dict):
        raise QuerySetError(f"queries[{index}] is not an object")
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise QuerySetError(f"queries[{index}] has no usable 'id'")
    query = raw.get("query")
    if not isinstance(query, str) or not query.strip():
        raise QuerySetError(f"query {case_id!r} has no usable 'query' text")
    course = raw.get("course")
    if course is not None and not isinstance(course, str):
        raise QuerySetError(f"query {case_id!r}: 'course' must be a string or null")
    expected = raw.get("expected_item_ids")
    if not isinstance(expected, list) or not all(
        isinstance(item, str) and item for item in expected
    ):
        raise QuerySetError(
            f"query {case_id!r}: 'expected_item_ids' must be a list of strings"
        )
    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        raise QuerySetError(f"query {case_id!r}: 'notes' must be a string")
    return QueryCase(
        id=case_id,
        query=query,
        course=course,
        expected_item_ids=tuple(expected),
        notes=notes,
    )


def load_query_set(path: Path) -> Lookup[QuerySet]:
    """Load + validate the labeled set at ``path``."""
    if not path.is_file():
        return Missing(reason=f"no query set at {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QuerySetError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise QuerySetError(f"{path}: top level is not an object")
    version = raw.get("set_version")
    if not isinstance(version, str) or not version.strip():
        raise QuerySetError(f"{path}: 'set_version' must be a non-empty string")
    census_date = raw.get("corpus_census_date")
    if census_date is not None and not isinstance(census_date, str):
        raise QuerySetError(f"{path}: 'corpus_census_date' must be a string or absent")
    raw_queries = raw.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise QuerySetError(f"{path}: 'queries' must be a non-empty list")
    cases = tuple(
        _parse_case(entry, index=index) for index, entry in enumerate(raw_queries)
    )
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise QuerySetError(f"{path}: duplicate query ids in {sorted(ids)}")
    return Found(
        value=QuerySet(
            set_version=version, queries=cases, corpus_census_date=census_date
        )
    )
