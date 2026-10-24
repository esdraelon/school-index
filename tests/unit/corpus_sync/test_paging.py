"""Regression tests: harvest paging to exhaustion (2026-10-24 fix).

A ``FakeAdapterTransport`` mimics the adapter's ``get_resources``
contract exactly — title order, limit applied AFTER filters with a
default of 200, course-filter union semantics (courseless rows
match every course filter), kind filter, ``student_id`` scoping —
so the paging client is proven against the contract, not against
a transport that happens to return everything at once. Every
fixture here straddles the 200 boundary: a fixture under the cap
proves nothing about paging.
"""

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from school_index.corpus_sync.client import (
    DEFAULT_LIMIT,
    McpResourceSource,
    PayloadError,
)
from school_index.corpus_sync.domain import ItemStatus
from school_index.corpus_sync.harvester import Harvester
from school_index.corpus_sync.lookup import Found
from school_index.corpus_sync.store import CorpusStore
from school_index.corpus_sync.strategies import build_strategies

SCRAPED = "2026-10-24T05:15:00+00:00"
COURSES = ["Algebra I", "Math 8", "Science"]
KINDS = ["DOCUMENT", "LINK", "NOTES", "TUTORIAL"]


def _item(
    *,
    url: str,
    title: str,
    course: str | None,
    kind: str,
    shared: bool = False,
) -> dict[str, Any]:
    return {
        "student": None
        if shared
        else {"id": "sam", "display_name": "Sam"},
        "course": None if course is None else {"name": course},
        "title": title,
        "kind": kind,
        "url": url,
        "source_kind": "AGENDA",
        "scraped_at": SCRAPED,
    }


class FakeAdapterTransport:
    """The adapter contract, as a transport (see module docstring)."""

    def __init__(
        self, items: list[dict[str, Any]], *, partition_drop: set[str] | None = None
    ) -> None:
        self.items = items
        self.partition_drop = partition_drop or set()
        self.calls: list[Mapping[str, Any]] = []
        self.returned_lengths: list[int] = []

    def __call__(self, tool: str, arguments: Mapping[str, Any]) -> object:
        assert tool == "get_resources"
        self.calls.append(dict(arguments))
        student_id = arguments["student_id"]
        visible = [
            item
            for item in self.items
            if item["student"] is None or item["student"]["id"] == student_id
        ]
        if "course_name" in arguments:
            wanted = arguments["course_name"]
            visible = [
                item
                for item in visible
                if item["course"] is None or item["course"]["name"] == wanted
            ]
        if "kind" in arguments:
            visible = [item for item in visible if item["kind"] == arguments["kind"]]
        if "limit" not in arguments and self.partition_drop:
            visible = [item for item in visible if item["url"] not in self.partition_drop]
        visible.sort(key=lambda item: str(item["title"]))
        limit = int(arguments.get("limit", DEFAULT_LIMIT))
        page = visible[: max(limit, 0)]
        self.returned_lengths.append(len(page))
        return {"result": page}


class _NoDownload:
    """Link-only fixtures must never touch the downloader."""

    def get(self, *, url: str) -> Any:
        raise AssertionError(f"unexpected download of {url}")


def _store(tmp_path: Path) -> CorpusStore:
    connection = sqlite3.connect(":memory:")
    return CorpusStore(
        connection_factory=lambda: connection, files_root=tmp_path / "files"
    )


def _harvester(transport: FakeAdapterTransport, store: CorpusStore) -> Harvester:
    source = McpResourceSource(url="http://unused", token="t", transport=transport)
    return Harvester(
        source=source,
        downloader=_NoDownload(),  # type: ignore[arg-type]
        strategies=build_strategies(),
        store=store,
    )


def _world_250(*, drift_title: str) -> list[dict[str, Any]]:
    """250 link-only items across 3 courses × 4 kinds + courseless.

    The drift item keeps one URL (its identity) while its title
    moves: that is the live drift mechanism — title change, same
    resource, new title position.
    """
    items: list[dict[str, Any]] = []
    for index in range(249):
        course: str | None = COURSES[index % len(COURSES)]
        if index % 21 == 0:
            course = None  # courseless: rides every course partition
        items.append(
            _item(
                url=f"https://example.com/resource-{index}",
                title=f"B{index:03d}",
                course=course,
                kind=KINDS[index % len(KINDS)],
                shared=index % 17 == 0,
            )
        )
    items.append(
        _item(
            url="https://example.com/drift-item",
            title=drift_title,
            course="Science",
            kind="LINK",
        )
    )
    return items


def test_paging_recovers_all_250_and_keeps_drifted_item(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Run 1: a 150-item world where the drift item sits inside the
    # first 200 by title — the old harvest would have kept it.
    small_world = _world_250(drift_title="A-drift")[:149]
    small_world.append(
        _item(
            url="https://example.com/drift-item",
            title="A-drift",
            course="Science",
            kind="LINK",
        )
    )
    transport = FakeAdapterTransport(small_world)
    report = _harvester(transport, store).run(student="sam")
    assert report.total == 150
    assert report.exhaustive is True

    # Run 2: the full 250-item world; the drift item's new title
    # sorts last — past the old 200 boundary. The old single-call
    # harvest would have tombstoned it as "vanished" here.
    big_world = _world_250(drift_title="Z-drift")
    titles = sorted(str(item["title"]) for item in big_world)
    assert titles.index("Z-drift") + 1 > 200  # the test earns its name
    transport.items = big_world
    transport.calls.clear()
    transport.returned_lengths.clear()
    report = _harvester(transport, store).run(student="sam")
    assert report.total == 250
    assert report.exhaustive is True
    assert report.tombstoned == 0
    # Partitions only (returned_lengths[0] is the 250-row probe leg):
    # every partition call provably returned fewer than the limit.
    assert max(transport.returned_lengths[1:]) < DEFAULT_LIMIT
    # 1 probe + 3 courses × 4 kinds, every call student-scoped
    # with the correct argument name.
    assert len(transport.calls) == 1 + len(COURSES) * len(KINDS)
    assert all(args.get("student_id") == "sam" for args in transport.calls)
    assert all("student" not in args for args in transport.calls)
    assert len(store.items_for_student(student="sam")) == 250
    drift = store.get_item(source_ref="sam:https://example.com/drift-item")
    assert isinstance(drift, Found)
    assert drift.value.status is ItemStatus.OK


def test_partition_at_default_limit_fails_loudly(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seed = FakeAdapterTransport(
        [_item(url="https://example.com/seed", title="Seed", course="Science", kind="LINK")]
    )
    _harvester(seed, store).run(student="sam")
    # One course × kind partition of exactly 200 rows: no finer
    # tool partition exists, so the source must refuse, and the
    # harvester must abort before tombstoning anything.
    block = [
        _item(
            url=f"https://example.com/block-{index}",
            title=f"P{index:03d}",
            course="Algebra I",
            kind="DOCUMENT",
        )
        for index in range(200)
    ]
    transport = FakeAdapterTransport(block)
    with pytest.raises(PayloadError, match="possibly truncated"):
        _harvester(transport, store).run(student="sam")
    kept = store.get_item(source_ref="sam:https://example.com/seed")
    assert isinstance(kept, Found)
    assert kept.value.status is ItemStatus.OK


def test_union_probe_disagreement_is_partial_not_complete() -> None:
    world = _world_250(drift_title="Z-drift")
    transport = FakeAdapterTransport(
        world, partition_drop={"https://example.com/resource-7"}
    )
    source = McpResourceSource(url="http://unused", token="t", transport=transport)
    listed = source.list_resources(student="sam")
    assert listed.probe_distinct == 250
    assert len(listed.resources) == 249
    assert listed.exhaustive is False


def test_probe_at_large_limit_fails_loudly() -> None:
    # A probe returning exactly PROBE_LIMIT rows may itself be
    # truncated: it proves nothing, so the source refuses rather
    # than partitioning off a doubtful total.
    one = _item(
        url="https://example.com/one", title="One", course="Science", kind="LINK"
    )
    transport = FakeAdapterTransport([one] * 100000)
    source = McpResourceSource(url="http://unused", token="t", transport=transport)
    with pytest.raises(PayloadError, match="probe itself may be truncated"):
        source.list_resources(student="sam")
    assert len(transport.calls) == 1  # refused before any partition


def test_all_courseless_world_pages_by_kind_only() -> None:
    # No course exists to partition on: the degenerate leg is one
    # kind-only partition per kind (course filter omitted), which
    # still carries every courseless row of that kind.
    world = [
        _item(url=f"https://example.com/c{i}", title=f"C{i:03d}", course=None, kind=k)
        for i, k in enumerate(["DOCUMENT", "LINK", "NOTES", "TUTORIAL"] * 5)
    ]
    transport = FakeAdapterTransport(world)
    source = McpResourceSource(url="http://unused", token="t", transport=transport)
    listed = source.list_resources(student="sam")
    assert listed.exhaustive is True
    assert len(listed.resources) == 20
    assert len(transport.calls) == 1 + 4
    assert all("course_name" not in args for args in transport.calls[1:])
