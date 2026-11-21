from app.rag import NOT_FOUND, build_user_prompt, cited_numbers


def test_cited_numbers_ignores_out_of_range():
    assert cited_numbers("Do X [1][3]. Then Y [9].", max_n=5) == [1, 3]


def test_index_sample(indexed):
    doc = indexed.db.get_document("doc1")
    assert doc["status"] == "indexed"
    assert doc["pages"] == 5 and doc["chunks"] >= 5


def test_answer_has_citation_with_page(indexed):
    r = indexed.answer("What is the phone extension for a fuel spill?")
    assert r.grounded
    assert r.citations and r.citations[0].page == 3
    assert "[1]" in r.answer


def test_refuses_when_nothing_relevant(indexed):
    r = indexed.answer("What is the baggage allowance for infants?")
    assert not r.grounded
    assert r.answer == NOT_FOUND
    assert r.citations == []


def test_uncited_llm_answer_is_rejected(indexed, monkeypatch):
    monkeypatch.setattr(indexed.llm, "complete", lambda s, u: ("Just trust me.", 10, 5))
    r = indexed.answer("What must be inspected after a bird strike?")
    assert r.answer == NOT_FOUND and not r.grounded


def test_queries_are_logged(indexed):
    indexed.answer("What must be inspected after a bird strike?")
    assert indexed.db.metrics()["queries"] == 1


def test_prompt_includes_history_and_sources(indexed):
    hits = indexed.retrieve("bird strike inspection")
    prompt = build_user_prompt("and then?", hits, [{"question": "bird strike?", "answer": "Inspect [1]"}])
    assert "Earlier in this conversation" in prompt and "[1] Sample SOP, page" in prompt


def test_page_offset_maps_to_original_pages(assistant, sample_pdf):
    assistant.db.add_document("doc2", "Sample SOP", "1.0", str(sample_pdf), page_offset=100)
    assistant.index_document("doc2", "Sample SOP", "1.0", sample_pdf, page_offset=100)
    r = assistant.answer("What is the phone extension for a fuel spill?")
    assert r.citations[0].page == 103
