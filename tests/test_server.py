import json
import time

import pytest
from fastapi.testclient import TestClient

import rag
import server
from conftest import make_pdf

HANDBOOK = make_pdf([
    ["Acme Robotics Employee Handbook", "Every employee gets 24 days of paid leave per year."],
    ["Remote work policy", "Employees may work remotely up to 3 days per week."],
])


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def similarity_search(self, query, k):
        self.queries.append(query)
        return self.chunks[:k]


class FakeChain:
    def __init__(self):
        self.parts = ["Employees get ", "**24 days** of paid leave [1]."]
        self.suggestions = '["How much leave do I get?", "Can I work remotely?", "What is Acme Robotics?"]'
        self.error = None
        self.prompts = []

    def stream(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        yield from self.parts

    def invoke(self, prompt):
        return self.suggestions


# No Gemini calls in tests
@pytest.fixture
def chain(monkeypatch):
    fake = FakeChain()
    monkeypatch.setattr(rag, "text_chain", lambda model: fake)
    monkeypatch.setattr(rag, "build_store", FakeStore)
    return fake


@pytest.fixture
def client(chain):
    server.registry.sessions.clear()
    with TestClient(server.app) as test_client:
        yield test_client
    server.registry.sessions.clear()


def upload(client, *files):
    files = files or (("handbook.pdf", HANDBOOK),)
    return client.post("/api/sessions", files=[("files", (name, data, "application/pdf")) for name, data in files])


def ask(client, session_id, question, history=()):
    res = client.post(f"/api/sessions/{session_id}/ask", json={"question": question, "history": list(history)})
    assert res.status_code == 200
    return [json.loads(line) for line in res.text.splitlines() if line]


def wait_for_suggestions(client, session_id):
    for _ in range(100):
        suggestions = client.get(f"/api/sessions/{session_id}").json()["suggestions"]
        if suggestions is not None:
            return suggestions
        time.sleep(0.02)
    raise AssertionError("suggestions never arrived")


def test_home_page_is_prerendered_with_security_headers(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "<title>ChatPDF" in res.text
    assert 'property="og:image"' in res.text
    assert "Drop your PDFs here" in res.text
    assert "default-src 'self'" in res.headers["content-security-policy"]
    assert client.get("/og.png").headers["content-type"] == "image/png"


def test_config_reports_limits(client):
    assert client.get("/api/config").json() == {"max_files": 3, "max_total_mb": 10, "max_pages": 100, "idle_minutes": 15}


@pytest.mark.parametrize("files, message", [
    ((("notes.txt", b"hello"),), "isn't a PDF"),
    (tuple((f"doc{i}.pdf", HANDBOOK) for i in range(4)), "up to 3 PDFs"),
    ((("blank.pdf", make_pdf([[]])),), "No text could be read"),
])
def test_bad_uploads_get_clear_errors(client, files, message):
    res = upload(client, *files)
    assert res.status_code == 400
    assert message in res.json()["detail"]
    assert server.registry.count() == 0


def test_corrupt_pdf_is_rejected(client):
    res = upload(client, ("broken.pdf", b"%PDF-1.4 this is not really a pdf"))
    assert res.status_code == 400
    assert "couldn't be opened" in res.json()["detail"] or "No text could be read" in res.json()["detail"]


def test_total_size_limit(client, monkeypatch):
    monkeypatch.setattr(rag, "MAX_TOTAL_MB", len(HANDBOOK) / 2 / (1024 * 1024))
    res = upload(client)
    assert res.status_code == 400
    assert "which is the limit" in res.json()["detail"]


def test_upload_suggest_and_stream_a_cited_answer(client, chain):
    res = upload(client)
    assert res.status_code == 201
    body = res.json()
    assert body["docs"] == [{"name": "handbook.pdf", "pages": 2}]
    session_id = body["session_id"]
    assert wait_for_suggestions(client, session_id) == ["How much leave do I get?", "Can I work remotely?", "What is Acme Robotics?"]

    events = ask(client, session_id, "How many leave days?")
    assert [e["type"] for e in events] == ["sources", "delta", "delta", "done"]
    first = events[0]["sources"][0]
    assert (first["id"], first["doc"], first["page"]) == (1, "handbook.pdf", 1)
    assert "24 days" in first["snippet"]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "Employees get **24 days** of paid leave [1]."
    assert "[1] (handbook.pdf, page 1)" in chain.prompts[-1]


def test_follow_up_questions_use_history(client, chain, monkeypatch):
    stores = []
    monkeypatch.setattr(rag, "build_store", lambda chunks: stores.append(FakeStore(chunks)) or stores[-1])
    session_id = upload(client).json()["session_id"]
    history = [{"role": "user", "content": "How many leave days?"}, {"role": "assistant", "content": "24 days [1]."}]
    ask(client, session_id, "And remote days?", history)
    assert stores[0].queries[-1] == "How many leave days?\nAnd remote days?"
    assert "User: How many leave days?" in chain.prompts[-1]


def test_rate_limited_answers_fall_back_to_the_other_model(client, monkeypatch):
    primary, fallback = FakeChain(), FakeChain()
    primary.error = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
    fallback.parts = ["From the fallback [1]."]
    chains = {rag.CHAT_MODEL: primary, rag.FALLBACK_CHAT_MODEL: fallback}
    monkeypatch.setattr(rag, "text_chain", lambda model: chains.get(model, primary))
    session_id = upload(client).json()["session_id"]
    events = ask(client, session_id, "Anything?")
    assert [e["type"] for e in events] == ["sources", "delta", "done"]
    assert events[1]["text"] == "From the fallback [1]."


def test_quota_errors_become_friendly_messages(client, chain):
    session_id = upload(client).json()["session_id"]
    chain.error = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
    events = ask(client, session_id, "Anything?")
    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "error"
    assert "quota" in events[-1]["message"]
    assert "RESOURCE_EXHAUSTED" not in events[-1]["message"]


def test_suggestions_fall_back_when_model_output_is_unusable(client, chain):
    chain.suggestions = "Sorry, I can't help with that."
    session_id = upload(client).json()["session_id"]
    assert wait_for_suggestions(client, session_id) == rag.FALLBACK_QUESTIONS


def test_deleted_session_returns_404(client):
    session_id = upload(client).json()["session_id"]
    assert client.delete(f"/api/sessions/{session_id}").status_code == 204
    assert client.get(f"/api/sessions/{session_id}").status_code == 404
    res = client.post(f"/api/sessions/{session_id}/ask", json={"question": "hi"})
    assert res.status_code == 404
    assert "expired" in res.json()["detail"]


def test_blank_question_is_rejected(client):
    session_id = upload(client).json()["session_id"]
    assert client.post(f"/api/sessions/{session_id}/ask", json={"question": "   "}).status_code == 422


def test_sessions_expire_when_abandoned_or_idle(client):
    abandoned, idle, active = (upload(client).json()["session_id"] for _ in range(3))
    now = time.time()
    server.registry.sessions[abandoned]["last_seen"] = now - server.ABANDONED_SECONDS - 1
    server.registry.sessions[idle]["last_used"] = now - server.IDLE_TIMEOUT_SECONDS - 1
    assert sorted(server.registry.flush_expired(now)) == sorted([abandoned, idle])
    assert client.get(f"/api/sessions/{active}").status_code == 200


def test_heartbeat_keeps_chat_alive_without_counting_as_activity(client):
    session_id = upload(client).json()["session_id"]
    entry = server.registry.sessions[session_id]
    entry["last_seen"] = entry["last_used"] = time.time() - 100
    client.get(f"/api/sessions/{session_id}")
    assert time.time() - entry["last_seen"] < 5
    assert time.time() - entry["last_used"] >= 100
