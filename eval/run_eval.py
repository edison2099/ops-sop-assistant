"""Run the test set and score the assistant (FSD section 9).

Usage:
  python -m eval.run_eval                       # use documents already indexed
  python -m eval.run_eval --index data/sample/sample_ground_ops_sop.pdf --title "Sample Ground Ops SOP"

Each test line: {"question", "keywords", "expected_page", "answerable"}
- accuracy:       answerable question, answer contains every expected keyword
- citation_match: answerable question, a citation points to expected_page
- refusal_rate:   unanswerable question, assistant says it can't find it
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from app.config import get_settings
from app.db import new_id
from app.main import build_assistant
from app.rag import NOT_FOUND

TARGETS = {"accuracy": 0.85, "citation_match": 0.90, "refusal_rate": 0.90}


def run(testset: Path, index_pdf: Path | None = None, title: str = "Sample SOP") -> dict:
    s = get_settings()
    a = build_assistant(s)
    if index_pdf:
        doc_id = new_id()
        a.db.add_document(doc_id, title, "1.0", str(index_pdf))
        doc = a.index_document(doc_id, title, "1.0", index_pdf)
        print(f"Indexed {title}: {doc['status']} ({doc['chunks']} chunks)")

    tests = [json.loads(line) for line in testset.read_text().splitlines() if line.strip()]
    details, latencies, costs = [], [], []
    for t in tests:
        r = a.answer(t["question"], session_id="eval")
        latencies.append(r.latency_ms)
        costs.append(r.cost_usd)
        refused = NOT_FOUND.lower() in r.answer.lower()
        pages = [c.page for c in r.citations]
        if t["answerable"]:
            correct = not refused and all(k.lower() in r.answer.lower() for k in t["keywords"])
            cited = t["expected_page"] in pages
        else:
            correct, cited = refused, None
        details.append({**t, "answer": r.answer, "cited_pages": pages, "correct": correct, "cited": cited})
        mark = "PASS" if correct else "FAIL"
        print(f"[{mark}] {t['question']}\n       -> {r.answer[:140]}  pages={pages}")

    ans = [d for d in details if d["answerable"]]
    unans = [d for d in details if not d["answerable"]]
    scores = {
        "accuracy": round(sum(d["correct"] for d in ans) / max(len(ans), 1), 3),
        "citation_match": round(sum(d["cited"] for d in ans) / max(len(ans), 1), 3),
        "refusal_rate": round(sum(d["correct"] for d in unans) / max(len(unans), 1), 3),
        "median_latency_ms": statistics.median(latencies) if latencies else 0,
        "avg_cost_usd": round(sum(costs) / max(len(costs), 1), 6),
    }
    run_id = a.db.save_eval(**scores, details=details)
    print(f"\nRun {run_id}")
    passed = True
    for k, v in scores.items():
        target = TARGETS.get(k)
        status = "" if target is None else ("  OK" if v >= target else f"  BELOW TARGET {target}")
        passed &= target is None or v >= target
        print(f"  {k:18} {v}{status}")
    scores["passed"] = passed
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--testset", type=Path, default=Path("eval/testset.jsonl"))
    p.add_argument("--index", type=Path, help="PDF to index before running")
    p.add_argument("--title", default="Sample Ground Ops SOP")
    args = p.parse_args()
    result = run(args.testset, args.index, args.title)
    sys.exit(0 if result["passed"] else 1)
