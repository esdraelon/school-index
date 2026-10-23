"""Unit tests: the structure-aware chunker (pure function proofs).

The stored-text contract under test is M2's: pages opened by
``[[page N]]`` marker lines. All expectations are computed from the
chunker's documented rules — pack whole pages to ~target, split
pages past max on sentences with ~overlap seeding.
"""

from school_index.embed_index.chunker import ChunkConfig, chunk_text

SMALL = ChunkConfig(target_chars=100, max_chars=140, overlap_chars=20)


def page(number: int, text: str) -> str:
    return f"[[page {number}]]\n{text}"


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("", item_id="i") == []
    assert chunk_text("   \n  ", item_id="i") == []


def test_single_short_page_is_one_chunk_with_page_heading() -> None:
    chunks = chunk_text(page(1, "Hello world."), item_id="i")
    assert len(chunks) == 1
    assert chunks[0].heading == "page 1"
    assert chunks[0].ordinal == 0
    assert chunks[0].text == "Hello world."


def test_pages_pack_to_target_then_flush() -> None:
    # Four 30-char pages, target 100: three pack (30+1+30+1+30=92),
    # the fourth would reach 123 > 100, so it starts a new chunk.
    text = "\n".join(page(n, "x" * 30) for n in range(1, 5))
    chunks = chunk_text(text, item_id="i", config=SMALL)
    assert len(chunks) == 2
    assert chunks[0].heading == "page 1"
    assert chunks[0].text == "\n".join(["x" * 30] * 3)
    assert chunks[1].heading == "page 4"
    assert chunks[1].ordinal == 1


def test_page_between_target_and_max_is_its_own_chunk() -> None:
    chunks = chunk_text(page(1, "y" * 120), item_id="i", config=SMALL)
    assert len(chunks) == 1
    assert len(chunks[0].text) == 120


def test_markerless_text_is_one_unheaded_segment() -> None:
    chunks = chunk_text("Just some prose without markers.", item_id="i")
    assert len(chunks) == 1
    assert chunks[0].heading is None


def test_blank_pages_are_skipped_and_page_numbers_preserved() -> None:
    text = "\n".join([page(1, "One."), page(2, "   "), page(3, "Three.")])
    chunks = chunk_text(text, item_id="i")
    assert len(chunks) == 1
    assert chunks[0].text == "One.\nThree."
    assert chunks[0].heading == "page 1"


def test_long_page_splits_on_sentences_with_overlap() -> None:
    sentences = [f"Sentence number {n} is here." for n in range(12)]
    big_page = " ".join(sentences)  # 12 * ~27 chars = ~320 > max 140
    chunks = chunk_text(page(5, big_page), item_id="i", config=SMALL)
    assert len(chunks) >= 3
    assert all(c.heading == "page 5" for c in chunks)
    assert all(len(c.text) <= SMALL.max_chars + SMALL.overlap_chars for c in chunks)
    # Every sentence survives somewhere in the split output.
    joined = " ".join(c.text for c in chunks)
    for sentence in sentences:
        assert sentence in joined
    # Overlap: consecutive pieces share text (the seed of piece n+1
    # is a tail of piece n).
    for first, second in zip(chunks, chunks[1:]):
        tail = first.text[-15:]
        assert tail.split()[-1] in second.text


def test_single_huge_sentence_hard_splits_at_word_boundaries() -> None:
    words = " ".join(f"word{n}" for n in range(120))  # one "sentence"
    chunks = chunk_text(page(1, words), item_id="i", config=SMALL)
    assert len(chunks) >= 4
    for chunk in chunks:
        assert not chunk.text.startswith(" ")
        assert len(chunk.text) <= SMALL.max_chars + SMALL.overlap_chars
    rejoined = " ".join(c.text for c in chunks).split()
    # No word is lost (words may repeat at overlap seams).
    assert set(rejoined) == set(words.split())


def test_chunking_is_deterministic() -> None:
    text = "\n".join(
        [page(1, "Alpha page."), page(2, " ".join(["Beta sentence."] * 40))]
    )
    first = chunk_text(text, item_id="i", config=SMALL)
    second = chunk_text(text, item_id="i", config=SMALL)
    assert first == second
    assert [c.ordinal for c in first] == list(range(len(first)))


def test_split_flushes_pending_buffer_first() -> None:
    text = "\n".join(
        [page(1, "Short page."), page(2, " ".join(["Long sentence here."] * 30))]
    )
    chunks = chunk_text(text, item_id="i", config=SMALL)
    assert chunks[0].text == "Short page."
    assert chunks[0].heading == "page 1"
    assert all(c.heading == "page 2" for c in chunks[1:])
