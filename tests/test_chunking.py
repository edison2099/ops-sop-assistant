from app.chunking import chunk_pages, detect_heading, split_sentences


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_chunks_keep_page_numbers_and_never_span_pages():
    pages = [(1, "Alpha sentence. " * 10), (2, "Beta sentence. " * 10)]
    chunks = chunk_pages(pages, chunk_words=50, overlap_words=10)
    assert {c.page for c in chunks} == {1, 2}
    assert all(("Alpha" in c.text) != ("Beta" in c.text) for c in chunks)


def test_chunk_size_and_overlap():
    text = " ".join(f"Sentence number {i} here." for i in range(200))
    chunks = chunk_pages([(1, text)], chunk_words=100, overlap_words=20)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 100 for c in chunks)
    # the last sentence of one chunk starts the next one
    last = chunks[0].text.split(". ")[-1]
    assert last.rstrip(".") in chunks[1].text


def test_overlap_must_be_smaller():
    import pytest

    with pytest.raises(ValueError):
        chunk_pages([(1, "x.")], chunk_words=10, overlap_words=10)


def test_heading_detection():
    assert detect_heading("2 Bird Strike Procedure")
    assert not detect_heading("After a reported bird strike, the engineer must inspect the aircraft.")
