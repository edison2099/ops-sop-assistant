from app.rag import system_prompt
from scripts.split_pdf import plan_sections


def test_plan_sections_keeps_parent_intro_pages_with_first_child():
    toc = [[1, "Book", 1], [2, "Safety", 3], [2, "Engine", 10], [3, "Precautions", 12], [3, "Fuel", 20]]
    s = plan_sections(toc, page_count=30, level=3)
    assert [(x["name"], x["first_page"], x["last_page"]) for x in s] == [
        ("Safety", 3, 9),
        ("Engine - Precautions", 10, 19),
        ("Engine - Fuel", 20, 30),
    ]


def test_plan_sections_without_bookmarks():
    assert plan_sections([], 12, 3) == [{"name": "Full document", "first_page": 1, "last_page": 12}]


def test_system_prompt_uses_role():
    p = system_prompt("a motorcycle service assistant")
    assert p.startswith("You are a motorcycle service assistant.")
    assert "I can't find this in the documents." in p


def test_local_embedder_uses_query_and_passage_modes(monkeypatch, settings):
    import sys
    import types

    import numpy as np

    from app.embeddings import LocalEmbedder, embed_query, get_embedder

    calls = []

    class FakeTextEmbedding:
        def __init__(self, name, cache_dir=None):
            calls.append(("init", name))

        def passage_embed(self, texts, batch_size=32):
            calls.append(("passage", len(texts)))
            return (np.ones(3) for _ in texts)

        def query_embed(self, texts):
            calls.append(("query", len(texts)))
            return (np.zeros(3) for _ in texts)

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    settings.embedding_provider = "local"
    settings.embedding_model = "text-embedding-3-small"  # OpenAI name left over -> falls back to bge
    e = get_embedder(settings)
    assert isinstance(e, LocalEmbedder) and e.model_name == "BAAI/bge-small-en-v1.5"
    assert e.embed(["a", "b"]) == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert embed_query(e, "q") == [0.0, 0.0, 0.0]
    assert calls == [("init", "BAAI/bge-small-en-v1.5"), ("passage", 2), ("query", 1)]


def test_threshold_defaults_per_provider(settings):
    settings.min_similarity = None
    for provider, expected in (("local", 0.6), ("openai", 0.3), ("hash", 0.2)):
        settings.embedding_provider = provider
        assert settings.similarity_threshold == expected
    settings.min_similarity = 0.45
    assert settings.similarity_threshold == 0.45
