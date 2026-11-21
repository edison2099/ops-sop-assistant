"""Split a large manual into one PDF per section, using the PDF's bookmarks.

Usage:
  python scripts/split_pdf.py manual.pdf out_folder --title-prefix "GSX-S750"

Splits at the deepest bookmark level up to --level (default 3). Pages between a parent bookmark
and its first child (e.g. a chapter's contents page) are kept with that first child.
Writes index.json with each file's title and original page range, which
scripts/bulk_upload.py uses so chatbot citations show the original page numbers.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf


def plan_sections(toc: list, page_count: int, level: int) -> list[dict]:
    """toc: [[lvl, title, page], ...] from PyMuPDF. Returns [{name, first_page, last_page}]."""
    toc = [e for e in toc if e[0] <= level]
    if not toc:
        return [{"name": "Full document", "first_page": 1, "last_page": page_count}]
    deepest = max(e[0] for e in toc)
    starts: list[tuple[str, int]] = []
    parents: dict[int, str] = {}
    carry: int | None = None  # start page of a parent whose pages go to its first child
    for i, (lvl, title, page) in enumerate(toc):
        parents[lvl] = title.strip()
        has_child = i + 1 < len(toc) and toc[i + 1][0] > lvl
        if lvl == 1 and deepest > 1:
            continue  # the whole-book bookmark
        if has_child:
            carry = page if carry is None else carry
            continue
        names = [parents[k] for k in sorted(parents) if 1 < k <= lvl] or [title.strip()]
        name = " - ".join(dict.fromkeys(names))  # drop repeats like "General Information - General Information"
        starts.append((name, carry if carry is not None else page))
        carry = None
    sections = []
    for (name, first), nxt in zip(starts, starts[1:] + [("", page_count + 1)], strict=True):
        if nxt[1] - 1 >= first:
            sections.append({"name": name, "first_page": first, "last_page": nxt[1] - 1})
    return sections


def split(src: Path, out: Path, prefix: str = "", level: int = 3) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(src)
    index = []
    for i, s in enumerate(plan_sections(doc.get_toc(), doc.page_count, level), start=1):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", s["name"]).strip("_")[:80]
        fname = f"{i:02d}_{safe}.pdf"
        part = pymupdf.open()
        part.insert_pdf(doc, from_page=s["first_page"] - 1, to_page=s["last_page"] - 1)
        title = f"{prefix} - {s['name']}" if prefix else s["name"]
        part.set_metadata({"title": title})
        part.save(out / fname, garbage=3, deflate=True)
        part.close()
        index.append({"file": fname, "title": title, **s, "pages": s["last_page"] - s["first_page"] + 1})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--title-prefix", default="")
    p.add_argument("--level", type=int, default=3)
    a = p.parse_args()
    idx = split(a.pdf, a.out, a.title_prefix, a.level)
    for e in idx:
        print(f"{e['file']:70} p{e['first_page']}-{e['last_page']}")
    print(f"\n{len(idx)} files written to {a.out}")
    sys.exit(0)
