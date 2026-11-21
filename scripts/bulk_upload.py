"""Upload every PDF in a folder to a running chatbot API.

Usage:
  python scripts/bulk_upload.py path/to/sections --api http://localhost:8000 --key YOUR_ADMIN_KEY

If the folder has an index.json (written by scripts/split_pdf.py), each file gets its title and
page_offset from it, so citations show the ORIGINAL manual page numbers. Otherwise the title is
taken from the file name.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--key", required=True, help="ADMIN_API_KEY")
    p.add_argument("--version", default="1.0")
    p.add_argument("--skip", nargs="*", default=[], help="file names to skip, e.g. 02-00_Table_of_Contents.pdf")
    args = p.parse_args()

    headers = {"X-Admin-Key": args.key}
    index_file = args.folder / "index.json"
    meta = {e["file"]: e for e in json.loads(index_file.read_text())} if index_file.exists() else {}

    pdfs = sorted(f for f in args.folder.glob("*.pdf") if f.name not in args.skip)
    if not pdfs:
        print("No PDFs found")
        return 1

    ids = {}
    for f in pdfs:
        m = meta.get(f.name, {})
        title = m.get("title") or f.stem.replace("_", " ")
        offset = m.get("first_page", 1) - 1
        with f.open("rb") as fh:
            r = requests.post(
                f"{args.api}/api/documents",
                headers=headers,
                files={"file": (f.name, fh, "application/pdf")},
                data={"title": title, "version": args.version, "page_offset": offset},
                timeout=300,
            )
        if not r.ok:
            print(f"FAILED  {f.name}: {r.status_code} {r.text[:200]}")
            continue
        ids[r.json()["doc_id"]] = f.name
        print(f"queued  {f.name}")

    # wait for indexing to finish
    pending = dict(ids)
    deadline = time.time() + 1800
    while pending and time.time() < deadline:
        time.sleep(2)
        for doc_id in list(pending):
            d = requests.get(f"{args.api}/api/documents/{doc_id}", headers=headers, timeout=30).json()
            if d["status"] != "processing":
                note = f"{d['chunks']} chunks" if d["status"] == "indexed" else d.get("error")
                print(f"{d['status']:8} {pending.pop(doc_id)} ({note})")

    print(f"\nDone: {len(ids) - len(pending)}/{len(pdfs)} files processed")
    return 0 if not pending else 1


if __name__ == "__main__":
    sys.exit(main())
