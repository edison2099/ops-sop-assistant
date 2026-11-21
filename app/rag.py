"""The RAG pipeline: index documents, then retrieve + generate grounded answers with citations."""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.chunking import chunk_pages
from app.config import Settings
from app.db import Database, new_id
from app.embeddings import Embedder
from app.ingest import extract_pages
from app.llm import LLM
from app.vectorstore import Hit, VectorStore

NOT_FOUND = "I can't find this in the documents."

SYSTEM_PROMPT_TEMPLATE = """You are {role}.
Answer ONLY using the numbered sources in the user's message.
Rules:
- Cite the source number in square brackets after every fact, e.g. [1] or [2][3].
- Do not use outside knowledge. Do not guess.
- If the sources do not contain the answer, reply exactly: "{not_found}"
- Keep answers short and practical. Use numbered steps for procedures.
- Copy numbers, torque values, part numbers and units exactly as written in the sources."""


def system_prompt(role: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(role=role, not_found=NOT_FOUND)


@dataclass
class Citation:
    n: int
    doc_id: str
    title: str
    page: int
    section_heading: str
    snippet: str
    similarity: float


@dataclass
class Answer:
    query_id: str
    answer: str
    grounded: bool
    citations: list[Citation] = field(default_factory=list)
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def build_user_prompt(question: str, hits: list[Hit], history: list[dict] | None = None) -> str:
    parts = []
    if history:
        parts.append("Earlier in this conversation:")
        for turn in history[-3:]:
            parts.append(f"Q: {turn.get('question', '')}\nA: {turn.get('answer', '')}")
        parts.append("")
    parts.append("Sources:")
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] {h.title}, page {h.page}\n{h.text}")
    parts.append(f"\nQuestion: {question}")
    return "\n".join(parts)


def cited_numbers(answer: str, max_n: int) -> list[int]:
    nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [n for n in nums if 1 <= n <= max_n]


class Assistant:
    def __init__(self, settings: Settings, db: Database, store: VectorStore, embedder: Embedder, llm: LLM):
        self.s, self.db, self.store, self.embedder, self.llm = settings, db, store, embedder, llm

    # ---------- indexing ----------
    def index_document(self, doc_id: str, title: str, version: str, pdf_path: Path, page_offset: int = 0) -> dict:
        try:
            # page_offset maps page 1 of this file to its page in the original manual,
            # so a section PDF that starts at manual page 438 has page_offset=437
            pages = [(n + page_offset, text) for n, text in extract_pages(pdf_path)]
            chunks = chunk_pages(pages, self.s.chunk_words, self.s.chunk_overlap_words)
            embeddings = self.embedder.embed([c.text for c in chunks])
            self.store.delete_document(doc_id)  # re-index replaces old chunks
            self.store.add(doc_id, title, version, chunks, embeddings)
            self.db.update_document(doc_id, status="indexed", error=None, pages=len(pages), chunks=len(chunks))
        except Exception as exc:
            self.db.update_document(doc_id, status="failed", error=str(exc)[:500])
        return self.db.get_document(doc_id)

    # ---------- answering ----------
    def retrieve(self, question: str, history: list[dict] | None = None) -> list[Hit]:
        # include the previous question so follow-ups like "and after that?" still find the right section
        query = question
        if history:
            query = f"{history[-1].get('question', '')} {question}"
        hits = self.store.search(self.embedder.embed([query])[0], self.s.top_k)
        return [h for h in hits if h.similarity >= self.s.min_similarity]

    def answer(self, question: str, session_id: str | None = None, history: list[dict] | None = None) -> Answer:
        start = time.perf_counter()
        query_id = new_id()
        hits = self.retrieve(question, history)

        if not hits:
            result = Answer(query_id, NOT_FOUND, grounded=False)
        else:
            text, tin, tout = self.llm.complete(
                system_prompt(self.s.assistant_role), build_user_prompt(question, hits, history)
            )
            text = text.strip()
            nums = cited_numbers(text, len(hits))
            grounded = bool(nums) and NOT_FOUND.lower() not in text.lower()
            citations = [
                Citation(
                    n=n,
                    doc_id=hits[n - 1].doc_id,
                    title=hits[n - 1].title,
                    page=hits[n - 1].page,
                    section_heading=hits[n - 1].section_heading,
                    snippet=hits[n - 1].text[:400],
                    similarity=hits[n - 1].similarity,
                )
                for n in nums
            ]
            if not grounded:
                # an answer with no citation is not trusted (FR-07): fall back to a refusal
                text, citations = NOT_FOUND, []
            cost = tin / 1e6 * self.s.llm_input_price_per_mtok + tout / 1e6 * self.s.llm_output_price_per_mtok
            result = Answer(query_id, text, grounded, citations, 0, tin, tout, round(cost, 6))

        result.latency_ms = int((time.perf_counter() - start) * 1000)
        self.db.log_query(
            query_id=query_id,
            session_id=session_id,
            question=question,
            chunk_ids=[h.chunk_id for h in hits],
            answer=result.answer,
            grounded=result.grounded,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        )
        return result
