"""Unit tests: the labeled query set loader.

The shipped set (benchmark/queries.json) is validated as data by
the last test — labels rotting against the corpus is a real failure
mode, and this at least pins the file's shape, version, size, and
empty-label budget on every run.
"""

import json
from pathlib import Path

import pytest

from school_index.benchmark.queryset import QuerySetError, load_query_set
from school_index.corpus_sync.lookup import Found, Missing

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_SET = REPO_ROOT / "benchmark" / "queries.json"


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _case(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "q01-x",
        "query": "how do I find the hypotenuse",
        "course": "Math 8",
        "expected_item_ids": ["item-a"],
        "notes": "a note",
    }
    base.update(overrides)
    return base


def test_missing_file_is_missing_not_an_error(tmp_path: Path) -> None:
    result = load_query_set(tmp_path / "absent.json")
    assert isinstance(result, Missing)
    assert "absent.json" in result.reason


def test_valid_set_loads(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "set_version": "9.9",
            "corpus_census_date": "2026-01-01",
            "queries": [
                _case(),
                _case(id="q02-y", course=None, expected_item_ids=[]),
            ],
        },
    )
    result = load_query_set(path)
    assert isinstance(result, Found)
    query_set = result.value
    assert query_set.set_version == "9.9"
    assert query_set.corpus_census_date == "2026-01-01"
    assert query_set.queries[0].expected_item_ids == ("item-a",)
    assert query_set.queries[1].course is None
    assert query_set.queries[1].expected_item_ids == ()


def test_malformed_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(QuerySetError, match="invalid JSON"):
        load_query_set(path)


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"queries": [_case()]},  # no set_version
        {"set_version": "  ", "queries": [_case()]},
        {"set_version": "1", "queries": []},
        {"set_version": "1", "queries": "nope"},
        {"set_version": "1", "corpus_census_date": 7, "queries": [_case()]},
    ],
)
def test_invalid_top_level_shapes_raise(tmp_path: Path, payload: object) -> None:
    with pytest.raises(QuerySetError):
        load_query_set(_write(tmp_path, payload))


@pytest.mark.parametrize(
    "case",
    [
        "not-an-object",
        _case(id=""),
        _case(id=3),
        _case(query=" "),
        _case(course=42),
        _case(expected_item_ids="item-a"),
        _case(expected_item_ids=["ok", ""]),
        _case(expected_item_ids=[1]),
        _case(notes=5),
    ],
)
def test_invalid_cases_raise(tmp_path: Path, case: object) -> None:
    with pytest.raises(QuerySetError):
        load_query_set(_write(tmp_path, {"set_version": "1", "queries": [case]}))


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    payload = {"set_version": "1", "queries": [_case(), _case(query="other text")]}
    with pytest.raises(QuerySetError, match="duplicate"):
        load_query_set(_write(tmp_path, payload))


def test_shipped_query_set_is_well_formed() -> None:
    result = load_query_set(SHIPPED_SET)
    assert isinstance(result, Found)
    query_set = result.value
    assert query_set.set_version == "1.0"
    # The plan calls for ~30 queries with at most ~4 absent-topic
    # (empty-label) probes; drift from that is a deliberate edit,
    # never an accident — this test is where the accident fails.
    assert 28 <= len(query_set.queries) <= 36
    empty = [q for q in query_set.queries if not q.expected_item_ids]
    assert len(empty) <= 4
    for case in query_set.queries:
        assert 1 <= len(case.expected_item_ids) <= 3 or not case.expected_item_ids
        assert case.notes  # every label carries its reasoning
