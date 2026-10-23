"""Unit tests: the embed pipeline over real on-disk stores.

Collaborators are the real repositories and the real chunker; the
only doubles are the embedding provider (deterministic hash fake)
and injected time, so delta-skip proofs can compare stamps.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from school_index.corpus_sync.domain import CorpusItem, ItemStatus
from school_index.corpus_sync.lookup import Found
from school_index.corpus_sync.store import CorpusStore
from school_index.doc_extract.domain import ExtractionOutcome, ExtractionState
from school_index.doc_extract.store import ExtractStore
from school_index.embed_index.domain import EmbedReport
from school_index.embed_index.runner import EmbedRunner, ModelMismatchError
from school_index.embed_index.store import BlobBackend, EmbedStore
from tests.unit.embed_index.fakes import FakeProvider

T1 = datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 10, 23, 16, 0, tzinfo=timezone.utc)


def make_item(source_ref: str, **overrides: object) -> CorpusItem:
    fields: dict[str, object] = {
        "id": hashlib.sha256(source_ref.encode("utf-8")).hexdigest(),
        "source_kind": "AGENDA",
        "source_ref": source_ref,
        "student": "sam",
        "course": "Algebra I",
        "title": "Item",
        "deep_link": source_ref.split(":", 1)[1],
        "origin_url": source_ref.split(":", 1)[1],
        "format": "pdf",
        "content_hash": "hash-1",
        "fetched_at": "2026-10-23T13:00:00+00:00",
        "status": ItemStatus.OK,
        "error": None,
    }
    fields.update(overrides)
    return CorpusItem(**fields)  # type: ignore[arg-type]


class World:
    """A temp corpus root wired with all three repositories."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.corpus = CorpusStore.open(root)
        self.extracts = ExtractStore.open(root)
        db_path = root / "corpus.db"

        def factory() -> sqlite3.Connection:
            return sqlite3.connect(db_path)

        self.embeds = EmbedStore(
            connection_factory=factory, backend=BlobBackend(dim=FakeProvider.dim)
        )

    def add(self, item: CorpusItem) -> CorpusItem:
        self.corpus.upsert_item(item)
        return item

    def extract(
        self,
        item: CorpusItem,
        *,
        outcome: ExtractionOutcome = ExtractionOutcome.EXTRACTED,
        text: str | None = "[[page 1]]\nSome real content here.",
    ) -> None:
        self.extracts.record_outcome(
            ExtractionState(
                item_id=item.id,
                outcome=outcome,
                extractor="pypdf" if outcome is not ExtractionOutcome.NOT_EXTRACTABLE else None,
                source_hash=item.content_hash,
                format=item.format,
                char_count=len(text) if text else 0,
                page_count=1,
                error=None,
                extracted_at="2026-10-23T14:00:00+00:00",
            ),
            text=text,
        )

    def run(
        self, *, at: datetime, provider: FakeProvider | None = None, rebuild: bool = False
    ) -> EmbedReport:
        runner = EmbedRunner(
            corpus_store=self.corpus,
            extract_store=self.extracts,
            embed_store=self.embeds,
            provider=provider or FakeProvider(),
            now=lambda: at,
        )
        return runner.run(rebuild=rebuild)


def populated_world(tmp_path: Path) -> tuple[World, dict[str, CorpusItem]]:
    world = World(tmp_path / "corpus")
    items = {
        "good": world.add(make_item("sam:https://example.com/good")),
        "blank": world.add(make_item("sam:https://example.com/blank")),
        "link": world.add(
            make_item("sam:https://example.com/link", format="external-link",
                      content_hash=None)
        ),
        "gone": world.add(
            make_item("sam:https://example.com/gone", status=ItemStatus.TOMBSTONED)
        ),
    }
    world.extract(items["good"])
    world.extract(items["blank"], outcome=ExtractionOutcome.EMPTY, text="")
    world.extract(
        items["link"], outcome=ExtractionOutcome.NOT_EXTRACTABLE, text=None
    )
    return world, items


def test_first_run_embeds_what_extraction_offers(tmp_path: Path) -> None:
    world, items = populated_world(tmp_path)
    report = world.run(at=T1)
    assert report.total == 4
    assert report.embedded == 2  # good + blank (0 chunks)
    assert report.not_embeddable == 1  # the link
    assert report.skipped == 0 and report.purged == 0 and report.failed == 0
    assert report.chunks_written == 1 and report.chunks_total == 1

    good_chunks = world.embeds.chunks_for_item(item_id=items["good"].id)
    assert [c.text for c in good_chunks] == ["Some real content here."]
    blank_state = world.embeds.get_embed_state(item_id=items["blank"].id)
    assert isinstance(blank_state, Found)
    assert blank_state.value.chunk_count == 0
    health = world.corpus.get_sync_state(source="embed-index")
    assert isinstance(health, Found)
    assert health.value.last_status == "ok"


def test_second_run_is_a_noop(tmp_path: Path) -> None:
    world, items = populated_world(tmp_path)
    world.run(at=T1)
    report = world.run(at=T2)
    assert report.embedded == 0
    assert report.skipped == 2
    state = world.embeds.get_embed_state(item_id=items["good"].id)
    assert isinstance(state, Found)
    assert state.value.embedded_at.startswith("2026-10-23T15:00")


def test_changed_extraction_reembeds_only_that_item(tmp_path: Path) -> None:
    world, items = populated_world(tmp_path)
    other = world.add(make_item("sam:https://example.com/other"))
    world.extract(other)
    world.run(at=T1)

    changed = make_item("sam:https://example.com/good", content_hash="hash-2")
    world.add(changed)
    world.extract(changed, text="[[page 1]]\nRevised content, now different.")

    report = world.run(at=T2)
    assert report.embedded == 1
    assert report.skipped == 2  # blank + other untouched
    chunks = world.embeds.chunks_for_item(item_id=items["good"].id)
    assert [c.text for c in chunks] == ["Revised content, now different."]
    other_state = world.embeds.get_embed_state(item_id=other.id)
    assert isinstance(other_state, Found)
    assert other_state.value.embedded_at.startswith("2026-10-23T15:00")


def test_tombstone_and_retraction_purge_the_index(tmp_path: Path) -> None:
    world, items = populated_world(tmp_path)
    world.run(at=T1)
    assert world.embeds.total_chunks() == 1

    tombstoned = make_item(
        "sam:https://example.com/good", status=ItemStatus.TOMBSTONED
    )
    world.add(tombstoned)
    report = world.run(at=T2)
    assert report.purged == 1
    assert world.embeds.total_chunks() == 0
    assert world.embeds.chunks_for_item(item_id=items["good"].id) == []


def test_orphaned_state_and_retracted_text_purge(tmp_path: Path) -> None:
    world, items = populated_world(tmp_path)
    # A state for an item the corpus has never heard of (a leftover).
    from school_index.embed_index.domain import Chunk, EmbedState

    world.embeds.replace_item_chunks(
        chunks=[Chunk(item_id="ghost", ordinal=0, heading=None, text="boo")],
        vectors=[(0.1,) * FakeProvider.dim],
        state=EmbedState(
            item_id="ghost",
            source_hash="h",
            model_id="fake-hash",
            chunk_count=1,
            embedded_at="2026-10-23T14:00:00+00:00",
        ),
    )
    first = world.run(at=T1)
    assert first.purged == 1  # the ghost's state is orphaned: purged
    assert world.embeds.chunks_for_item(item_id="ghost") == []

    # Retraction: the good item's extraction fails on re-extract —
    # its text row is deleted, so the index must drop it too.
    world.extracts.record_outcome(
        ExtractionState(
            item_id=items["good"].id,
            outcome=ExtractionOutcome.FAILED,
            extractor="pypdf",
            source_hash="hash-2",
            format="pdf",
            char_count=0,
            page_count=None,
            error="PdfReadError: broken",
            extracted_at="2026-10-23T15:30:00+00:00",
        ),
        text=None,
    )
    report = world.run(at=T2)
    assert report.not_embeddable == 2  # link + retracted good
    assert report.purged == 1
    assert world.embeds.chunks_for_item(item_id=items["good"].id) == []


class OtherModelProvider(FakeProvider):
    model_id = "other-model"


def test_model_change_requires_rebuild(tmp_path: Path) -> None:
    world, _items = populated_world(tmp_path)
    world.run(at=T1)
    with pytest.raises(ModelMismatchError, match="--rebuild"):
        world.run(at=T2, provider=OtherModelProvider())
    report = world.run(at=T2, provider=OtherModelProvider(), rebuild=True)
    assert report.embedded == 2
    states = world.embeds.embed_states()
    assert {s.model_id for s in states} == {"other-model"}


class ExplodingProvider(FakeProvider):
    def embed_documents(self, texts):  # type: ignore[override]
        raise RuntimeError("onnx said no")


def test_provider_failure_is_contained_per_item(tmp_path: Path) -> None:
    world, items = populated_world(tmp_path)
    report = world.run(at=T1, provider=ExplodingProvider())
    # Two embeddable items, both fail; the run completes and records.
    assert report.failed == 2
    assert report.embedded == 0
    assert report.failure_tally() == {"RuntimeError: onnx said no": 2}
    health = world.corpus.get_sync_state(source="embed-index")
    assert isinstance(health, Found)
    assert health.value.last_status == "partial"
    # Nothing was written for the failed item, so a healthy provider
    # picks it up on the next run.
    report = world.run(at=T2)
    assert report.embedded == 2
    assert isinstance(
        world.embeds.get_embed_state(item_id=items["good"].id), Found
    )
