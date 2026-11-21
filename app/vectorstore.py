"""ChromaDB wrapper: stores chunks with document name and page metadata."""

from dataclasses import dataclass

import chromadb

from app.chunking import Chunk
from app.config import Settings


@dataclass
class Hit:
    chunk_id: str
    text: str
    doc_id: str
    title: str
    page: int
    section_heading: str
    similarity: float  # 1.0 = identical meaning, 0 = unrelated


class VectorStore:
    def __init__(self, settings: Settings, client: chromadb.ClientAPI | None = None):
        self.client = client or chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.col = self.client.get_or_create_collection(settings.collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, doc_id: str, title: str, version: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.col.upsert(
            ids=[f"{doc_id}:{c.index}" for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "title": title,
                    "version": version,
                    "page": c.page,
                    "section_heading": c.section_heading,
                }
                for c in chunks
            ],
        )

    def search(self, embedding: list[float], k: int) -> list[Hit]:
        if self.col.count() == 0:
            return []
        r = self.col.query(
            query_embeddings=[embedding],
            n_results=min(k, self.col.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, text, meta, dist in zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0], strict=True
        ):
            hits.append(
                Hit(
                    chunk_id=cid,
                    text=text,
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                    page=int(meta["page"]),
                    section_heading=meta.get("section_heading", ""),
                    similarity=round(1.0 - float(dist), 4),
                )
            )
        return hits

    def delete_document(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self.col.count()
