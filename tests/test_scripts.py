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
