# ChatPDF

Chat with your PDFs: upload up to 3 documents and ask questions. Answers stream in live, cite the exact page they came from, and come only from what's inside your files.

Live at **https://chatpdf.siddharthranjan.app**

## Features

- Drag-and-drop upload that indexes automatically
- Streaming answers with clickable citations showing the source passage and page
- Follow-up questions that understand the conversation so far
- AI-suggested starter questions for each set of documents
- Light and dark themes, designed for phones as well as desktops
- Private: documents are kept only in server memory and deleted when you close the tab or stay idle

## How it works

| Part | What it does |
| --- | --- |
| `web/` | Static frontend (HTML, CSS, vanilla JS). No build step. |
| `server.py` | FastAPI app: upload, streaming answers (NDJSON), session heartbeat and cleanup. Also serves `web/`. |
| `rag.py` | PDF text extraction with page numbers, chunking, FAISS retrieval and Gemini prompts. |

Retrieval uses Gemini embeddings and an in-memory FAISS index per chat. Chats are deleted when the page stops sending heartbeats (tab closed) or after a period without questions.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
echo "GOOGLE_API_KEY=your-key" > .env
uvicorn server:app --reload --port 8501
```

Open http://localhost:8501. Run the tests with `pytest`; they use fakes, so no API key or quota is needed.

## Configuration

Set these in `.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `CHAT_MODEL` | `gemini-3.5-flash-lite` | Model that writes answers |
| `FALLBACK_CHAT_MODEL` | `gemma-4-26b-a4b-it` | Answers with this when `CHAT_MODEL` hits its rate limit |
| `SUGGESTION_MODEL` | `gemma-4-26b-a4b-it` | Model that writes starter questions |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model |
| `MAX_FILES` / `MAX_TOTAL_MB` / `MAX_PAGES` | `3` / `10` / `100` | Upload limits |
| `IDLE_TIMEOUT_MINUTES` | `15` | Delete a chat after this long without questions |
| `ABANDONED_SECONDS` | `600` | Delete a chat this long after its tab stops sending heartbeats |
| `MAX_SESSIONS` | `20` | Concurrent chats kept in memory |

## Deploy

`docker compose up -d --build` runs the app behind Caddy, which provides HTTPS. Edit the domain in `Caddyfile`. The app must run as a single process because chats live in memory.
