# Ops SOP Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers operations questions from SOP and manual PDFs, with a **page citation on every answer**. If the answer isn't in the documents, it says so instead of guessing.

```
Indexing (once per document)
  PDF ──► parse (keep page numbers) ──► chunk ──► embed ──► ChromaDB

Answering (every question)
  Chat UI ──► FastAPI ──► embed question ──► top 5 chunks (similarity ≥ threshold)
                │                                   │ none relevant? → "I can't find this in the documents."
                └──► LLM (answer ONLY from sources, cite [n]) ──► answer + document/page citations
```

## Features

- PDF upload, chunking (~600 tokens, 100 overlap, never mid-sentence or across pages), embeddings in ChromaDB
- Grounded answers: strict system prompt, relevance threshold, and **uncited answers are rejected**
- Citations with document title, page, section heading and snippet
- Follow-up questions use the last 3 turns of the conversation
- Feedback (👍/👎), query logs with latency, tokens and cost (SQLite)
- Admin API protected by `X-Admin-Key`: upload, list, re-index, delete, metrics
- Evaluation script that scores accuracy, citation match and correct refusals
- Swappable providers: Anthropic or OpenAI for the LLM; **free local embeddings** (default) or OpenAI embeddings; plus offline modes (no API key) for tests and demos
- Docker, docker-compose, GitHub Actions CI

## Quick start

### 1. Offline demo (no API key)

```bash
pip install -r requirements-dev.txt -r requirements-ui.txt
python scripts/make_sample_pdf.py           # fictional sample SOP -> data/sample/
cp .env.example .env
# in .env set: LLM_PROVIDER=fake  EMBEDDING_PROVIDER=hash  MIN_SIMILARITY=0.2
uvicorn app.main:app --reload --port 8000   # API docs at http://localhost:8000/docs
streamlit run ui/streamlit_app.py           # UI at http://localhost:8501
```

In the UI, open **Admin**, enter your `ADMIN_API_KEY`, and upload `data/sample/sample_ground_ops_sop.pdf`. Then ask questions on the **Ask** tab.

Offline mode uses keyword search and a stand-in "LLM" that quotes the best-matching sentence. It proves the pipeline works; it does not really understand text.

### 2. Real mode (only a Claude API key needed)

Embeddings run **free on your computer** by default (`EMBEDDING_PROVIDER=local`, model `BAAI/bge-small-en-v1.5`
via fastembed, CPU only). The model (~130 MB) downloads once on first use into `data/models/`.

In `.env` set:

```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-5-5
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_PROVIDER=local
```

Get the key at console.anthropic.com → API Keys (API credit is billed in the Console, separate from a claude.ai
plan). Prefer OpenAI? Set `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`, and optionally
`EMBEDDING_PROVIDER=openai` with `EMBEDDING_MODEL=text-embedding-3-small`.

**Changing the embedding provider or model means re-indexing:** delete `data/chroma/` and upload the documents again.
The relevance cutoff (`MIN_SIMILARITY`) has a sensible default per provider; tune it with the evaluation script.

### 3. Docker

```bash
docker compose up --build    # API on :8000, UI on :8501
```

## Load a large manual (e.g. a service manual)

Split it into one PDF per section using its bookmarks, then upload the whole folder:

```bash
python scripts/split_pdf.py manual.pdf sections/ --title-prefix "GSX-S750"
python scripts/bulk_upload.py sections/ --api http://localhost:8000 --key $ADMIN_API_KEY \
  --skip 02_Table_of_Contents.pdf
```

`split_pdf.py` writes `index.json` with each section's original page range, and `bulk_upload.py` passes it on as
`page_offset`, so citations show the **original manual page**, not the page inside the section file.

Set the assistant's role in `.env` to match your documents:

```
ASSISTANT_ROLE=a motorcycle service assistant for the Suzuki GSX-S750 service manual
```

Manuals are copyrighted. Keep them in `data/` (git-ignored) and out of public repos and demos.

## API

| Method | Endpoint | Purpose | Auth |
| --- | --- | --- | --- |
| POST | `/api/documents` | Upload a PDF (`file`, `title`, `version`, `page_offset`) and index it | Admin |
| GET | `/api/documents` | List documents and their status | Admin |
| GET | `/api/documents/{id}` | One document's status | Admin |
| POST | `/api/documents/{id}/reindex` | Re-index a document | Admin |
| DELETE | `/api/documents/{id}` | Delete a document and its chunks | Admin |
| POST | `/api/chat` | `{question, session_id?, history?}` → answer + citations | — |
| POST | `/api/feedback` | `{query_id, rating: 1 or -1, comment?}` | — |
| GET | `/api/metrics` | Query count, grounded rate, latency, ratings, cost | Admin |
| GET | `/health` | Health check | — |

```bash
curl -X POST localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"question": "How far must vehicles stay from a fuel spill?"}'
```

```json
{
  "answer": "Keep all vehicles and ground power units at least 15 metres away from the spill. [1]",
  "grounded": true,
  "citations": [{"n": 1, "title": "Sample SOP", "page": 3, "section_heading": "3 Fuel Spill Procedure", "...": "..."}]
}
```

## Evaluation

```bash
python -m eval.run_eval --index data/sample/sample_ground_ops_sop.pdf
```

The script scores against `eval/testset.jsonl` (answerable + unanswerable questions):

| Metric | Meaning | Target |
| --- | --- | --- |
| accuracy | Answer contains the expected facts | ≥ 85% |
| citation_match | A citation points to the expected page | ≥ 90% |
| refusal_rate | Unanswerable questions are refused | ≥ 90% |

Offline mode currently scores 100% on citation match and refusals. Accuracy targets apply to a real LLM. Re-run after changing chunking, prompts, threshold or model, and compare runs.

## Project layout

```
app/
  config.py       settings from .env
  ingest.py       PDF -> pages (PyMuPDF)
  chunking.py     sentence-aware chunks with overlap and page numbers
  embeddings.py   OpenAI or offline hash embeddings
  vectorstore.py  ChromaDB wrapper
  llm.py          Anthropic, OpenAI or offline fake LLM
  rag.py          retrieval, grounding prompt, citation check, logging
  db.py           SQLite: documents, query logs, feedback, eval runs
  main.py         FastAPI endpoints
ui/streamlit_app.py   chat + admin UI
eval/                 test set and evaluation runner
scripts/              sample PDF generator, PDF splitter, bulk uploader
tests/                pytest suite (runs offline)
```

## Tests

```bash
pytest -q
```

## Data note

Use only public documents (e.g. ICAO or CAAM publications) or the generated sample. The sample SOP is **fictional**. Do not upload internal or confidential company documents to a public demo.

## Roadmap

- Hybrid search (keyword + vector) and re-ranking
- OCR for scanned PDFs
- DOCX and web page sources
- User login (Google sign-in) and roles
- Deploy to Google Cloud Run with secrets in Secret Manager
