"""HTTP API and static site for ChatPDF.

Indexes live in memory, one per chat. A chat is deleted when its page stops
sending heartbeats (tab closed) or after IDLE_TIMEOUT_MINUTES without questions,
so run a single worker.
"""
import ctypes
import gc
import json
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag

IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_MINUTES", "15")) * 60
# The page pings every 30s; missing pings for this long means the tab is gone
ABANDONED_SECONDS = int(os.getenv("ABANDONED_SECONDS", "180"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "20"))
MAX_CONCURRENT_INDEXING = int(os.getenv("MAX_CONCURRENT_INDEXING", "2"))
JANITOR_INTERVAL_SECONDS = 30
WEB_DIR = Path(__file__).parent / "web"


def release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class Registry:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}

    def add(self, store, docs):
        session_id = secrets.token_urlsafe(18)
        now = time.time()
        entry = {"store": store, "docs": docs, "suggestions": None, "last_used": now, "last_seen": now}
        with self.lock:
            self.sessions[session_id] = entry
        return session_id, entry

    def get(self, session_id, activity=True):
        with self.lock:
            entry = self.sessions.get(session_id)
            if entry is not None:
                entry["last_seen"] = time.time()
                if activity:
                    entry["last_used"] = entry["last_seen"]
            return entry

    def remove(self, session_id):
        with self.lock:
            removed = self.sessions.pop(session_id, None)
        if removed:
            release_memory()
        return removed is not None

    def flush_expired(self, now=None):
        now = now or time.time()
        with self.lock:
            expired = [
                sid for sid, e in self.sessions.items()
                if now - e["last_used"] > IDLE_TIMEOUT_SECONDS or now - e["last_seen"] > ABANDONED_SECONDS
            ]
            for sid in expired:
                del self.sessions[sid]
        if expired:
            release_memory()
        return expired

    def count(self):
        with self.lock:
            return len(self.sessions)


registry = Registry()
indexing_slots = threading.BoundedSemaphore(MAX_CONCURRENT_INDEXING)


def janitor():
    while True:
        time.sleep(JANITOR_INTERVAL_SECONDS)
        registry.flush_expired()


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=janitor, name="janitor", daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    )
    return response


def get_entry(session_id, activity=True):
    entry = registry.get(session_id, activity)
    if entry is None:
        raise HTTPException(404, "This chat has expired. Upload your PDFs again to continue.")
    return entry


@app.get("/api/config")
def config():
    return {
        "max_files": rag.MAX_FILES,
        "max_total_mb": rag.MAX_TOTAL_MB,
        "max_pages": rag.MAX_PAGES,
        "idle_minutes": IDLE_TIMEOUT_SECONDS // 60,
    }


@app.get("/healthz")
def health():
    return {"ok": True, "sessions": registry.count()}


@app.post("/api/sessions", status_code=201)
def create_session(files: list[UploadFile] = File(...)):
    if len(files) > rag.MAX_FILES:
        raise HTTPException(400, f"You can upload up to {rag.MAX_FILES} PDFs at a time.")
    # Read at most one byte past the limit, so oversized uploads aren't held in memory
    limit = int(rag.MAX_TOTAL_MB * 1024 * 1024)
    payload, total = [], 0
    for upload in files:
        data = upload.file.read(limit - total + 1)
        total += len(data)
        payload.append((upload.filename or "document.pdf", data))
        if total > limit:
            break

    try:
        rag.validate_files(payload)
    except rag.UploadError as e:
        raise HTTPException(400, str(e))

    if registry.count() >= MAX_SESSIONS:
        registry.flush_expired()
        if registry.count() >= MAX_SESSIONS:
            raise HTTPException(503, "ChatPDF is busy right now. Please try again in a few minutes.")
    if not indexing_slots.acquire(timeout=30):
        raise HTTPException(503, "ChatPDF is busy right now. Please try again in a minute.")
    try:
        pages, docs = rag.extract_pages(payload)
        store = rag.build_store(rag.chunk_pages(pages))
    except rag.UploadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        print(f"indexing failed: {e!r}", flush=True)
        raise HTTPException(502, rag.friendly_error(e))
    finally:
        indexing_slots.release()
        del payload

    session_id, entry = registry.add(store, docs)

    def suggest():
        entry["suggestions"] = rag.generate_suggestions(pages)

    threading.Thread(target=suggest, name="suggestions", daemon=True).start()
    return {"session_id": session_id, "docs": docs}


# Also the page's heartbeat: keeps the chat alive while its tab is open
@app.get("/api/sessions/{session_id}")
def read_session(session_id: str):
    entry = get_entry(session_id, activity=False)
    return {"docs": entry["docs"], "suggestions": entry["suggestions"]}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Message] = Field(default_factory=list, max_length=40)


@app.post("/api/sessions/{session_id}/ask")
def ask(session_id: str, body: Question):
    entry = get_entry(session_id)
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "Please type a question.")
    history = [m.model_dump() for m in body.history]

    def events():
        # One JSON object per line: sources, then text deltas, then done or error
        try:
            for kind, value in rag.stream_answer(entry["store"], question, history):
                entry["last_used"] = entry["last_seen"] = time.time()
                if kind == "sources":
                    yield json.dumps({"type": "sources", "sources": value}) + "\n"
                else:
                    yield json.dumps({"type": "delta", "text": value}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            print(f"answer failed: {e!r}", flush=True)
            yield json.dumps({"type": "error", "message": rag.friendly_error(e)}) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    registry.remove(session_id)
    return Response(status_code=204)


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
