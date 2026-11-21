"""SQLite storage for documents, query logs, feedback and evaluation runs."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    version TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL,           -- processing | indexed | failed
    error TEXT,
    pages INTEGER DEFAULT 0,
    page_offset INTEGER DEFAULT 0,
    chunks INTEGER DEFAULT 0,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS query_logs (
    query_id TEXT PRIMARY KEY,
    session_id TEXT,
    question TEXT NOT NULL,
    chunk_ids TEXT NOT NULL,        -- JSON list
    answer TEXT NOT NULL,
    grounded INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    query_id TEXT PRIMARY KEY REFERENCES query_logs(query_id),
    rating INTEGER NOT NULL,        -- 1 = up, -1 = down
    comment TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    accuracy REAL, citation_match REAL, refusal_rate REAL,
    median_latency_ms REAL, avg_cost_usd REAL,
    details TEXT,
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, path: Path):
        self.path = path
        with self.conn() as c:
            c.executescript(SCHEMA)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(documents)")}
            if "page_offset" not in cols:  # databases created before page offsets existed
                c.execute("ALTER TABLE documents ADD COLUMN page_offset INTEGER DEFAULT 0")

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # documents
    def add_document(self, doc_id: str, title: str, version: str, file_path: str, page_offset: int = 0) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO documents (doc_id,title,version,file_path,status,page_offset,uploaded_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (doc_id, title, version, file_path, "processing", page_offset, now()),
            )

    def update_document(self, doc_id: str, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE documents SET {cols} WHERE doc_id=?", (*fields.values(), doc_id))

    def get_document(self, doc_id: str) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")]

    def delete_document(self, doc_id: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))

    # queries and feedback
    def log_query(self, **row) -> None:
        row["chunk_ids"] = json.dumps(row["chunk_ids"])
        row["grounded"] = int(row["grounded"])
        row["created_at"] = now()
        cols = ",".join(row)
        with self.conn() as c:
            c.execute(f"INSERT INTO query_logs ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))

    def query_exists(self, query_id: str) -> bool:
        with self.conn() as c:
            return c.execute("SELECT 1 FROM query_logs WHERE query_id=?", (query_id,)).fetchone() is not None

    def add_feedback(self, query_id: str, rating: int, comment: str | None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO feedback (query_id,rating,comment,created_at) VALUES (?,?,?,?)",
                (query_id, rating, comment, now()),
            )

    def metrics(self) -> dict:
        with self.conn() as c:
            q = c.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(cost_usd),0) cost, COALESCE(AVG(latency_ms),0) lat, "
                "COALESCE(AVG(grounded),0) grounded FROM query_logs"
            ).fetchone()
            f = c.execute("SELECT SUM(rating=1) up, SUM(rating=-1) down FROM feedback").fetchone()
        return {
            "queries": q["n"],
            "total_cost_usd": round(q["cost"], 4),
            "avg_latency_ms": round(q["lat"]),
            "grounded_rate": round(q["grounded"], 3),
            "thumbs_up": f["up"] or 0,
            "thumbs_down": f["down"] or 0,
        }

    # evaluation
    def save_eval(self, **row) -> str:
        run_id = new_id()
        row["details"] = json.dumps(row.get("details", []))
        with self.conn() as c:
            c.execute(
                "INSERT INTO eval_runs (run_id,accuracy,citation_match,refusal_rate,"
                "median_latency_ms,avg_cost_usd,details,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    row["accuracy"],
                    row["citation_match"],
                    row["refusal_rate"],
                    row["median_latency_ms"],
                    row["avg_cost_usd"],
                    row["details"],
                    now(),
                ),
            )
        return run_id
