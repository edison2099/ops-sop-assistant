"""FastAPI app. Run: uvicorn app.main:app --reload"""

import shutil
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.db import Database, new_id
from app.embeddings import get_embedder
from app.llm import get_llm
from app.rag import Assistant
from app.vectorstore import VectorStore


class Turn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    session_id: str | None = None
    history: list[Turn] = []


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(description="1 = thumbs up, -1 = thumbs down")
    comment: str | None = Field(default=None, max_length=1000)


def build_assistant(s: Settings) -> Assistant:
    return Assistant(s, Database(s.sqlite_path), VectorStore(s), get_embedder(s), get_llm(s))


def create_app(settings: Settings | None = None, assistant: Assistant | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="Ops SOP Assistant", version="0.1.0")
    app.state.assistant = assistant or build_assistant(s)

    def get_assistant() -> Assistant:
        return app.state.assistant

    def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
        if x_admin_key != s.admin_api_key:
            raise HTTPException(status_code=401, detail="Admin key required (header X-Admin-Key)")

    @app.get("/health")
    def health(a: Assistant = Depends(get_assistant)):
        return {"status": "ok", "chunks": a.store.count()}

    # ---------- documents (admin) ----------
    @app.post("/api/documents", dependencies=[Depends(require_admin)], status_code=202)
    def upload_document(
        background: BackgroundTasks,
        file: UploadFile = File(...),
        title: str = Form(...),
        version: str = Form("1.0"),
        page_offset: int = Form(0, ge=0, description="Original page number of this file's page 1, minus 1"),
        a: Assistant = Depends(get_assistant),
    ):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are supported")
        doc_id = new_id()
        dest = s.uploads_dir / f"{doc_id}.pdf"
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        if dest.stat().st_size > s.max_upload_mb * 1024 * 1024:
            dest.unlink()
            raise HTTPException(413, f"File is larger than {s.max_upload_mb} MB")
        a.db.add_document(doc_id, title, version, str(dest), page_offset)
        background.add_task(a.index_document, doc_id, title, version, dest, page_offset)
        return {"doc_id": doc_id, "status": "processing"}

    @app.get("/api/documents", dependencies=[Depends(require_admin)])
    def list_documents(a: Assistant = Depends(get_assistant)):
        return a.db.list_documents()

    @app.get("/api/documents/{doc_id}", dependencies=[Depends(require_admin)])
    def get_document(doc_id: str, a: Assistant = Depends(get_assistant)):
        doc = a.db.get_document(doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        return doc

    @app.post("/api/documents/{doc_id}/reindex", dependencies=[Depends(require_admin)], status_code=202)
    def reindex(doc_id: str, background: BackgroundTasks, a: Assistant = Depends(get_assistant)):
        doc = a.db.get_document(doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        a.db.update_document(doc_id, status="processing")
        background.add_task(
            a.index_document, doc_id, doc["title"], doc["version"], Path(doc["file_path"]), doc.get("page_offset") or 0
        )
        return {"doc_id": doc_id, "status": "processing"}

    @app.delete("/api/documents/{doc_id}", dependencies=[Depends(require_admin)])
    def delete_document(doc_id: str, a: Assistant = Depends(get_assistant)):
        doc = a.db.get_document(doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        a.store.delete_document(doc_id)
        a.db.delete_document(doc_id)
        Path(doc["file_path"]).unlink(missing_ok=True)
        return {"deleted": doc_id}

    # ---------- chat (users) ----------
    @app.post("/api/chat")
    def chat(req: ChatRequest, a: Assistant = Depends(get_assistant)):
        result = a.answer(req.question, req.session_id, [t.model_dump() for t in req.history])
        return {
            "query_id": result.query_id,
            "answer": result.answer,
            "grounded": result.grounded,
            "citations": [c.__dict__ for c in result.citations],
            "latency_ms": result.latency_ms,
        }

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest, a: Assistant = Depends(get_assistant)):
        if req.rating not in (1, -1):
            raise HTTPException(400, "rating must be 1 or -1")
        if not a.db.query_exists(req.query_id):
            raise HTTPException(404, "Unknown query_id")
        a.db.add_feedback(req.query_id, req.rating, req.comment)
        return {"ok": True}

    @app.get("/api/metrics", dependencies=[Depends(require_admin)])
    def metrics(a: Assistant = Depends(get_assistant)):
        return a.db.metrics()

    return app


app = create_app()
