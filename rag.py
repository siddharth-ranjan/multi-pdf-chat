"""PDF indexing, retrieval and Gemini calls. The web layer lives in server.py."""
import io
import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
# Free-tier daily quotas differ a lot per model: flash-lite is fast, and Gemma's much larger
# (but slower) quota takes over whenever flash-lite is rate limited
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")
FALLBACK_CHAT_MODEL = os.getenv("FALLBACK_CHAT_MODEL", "gemma-4-26b-a4b-it")
SUGGESTION_MODEL = os.getenv("SUGGESTION_MODEL", CHAT_MODEL)

# Upload limits keep memory use predictable on a small server
MAX_FILES = int(os.getenv("MAX_FILES", "3"))
MAX_TOTAL_MB = float(os.getenv("MAX_TOTAL_MB", "10"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "100"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "500000"))

# Small chunks keep citations pointing at the right page
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 5
SNIPPET_CHARS = 280
HISTORY_MESSAGES = 6

# Used only if Gemini can't generate document-specific suggestions
FALLBACK_QUESTIONS = [
    "Summarize these documents in 5 bullet points",
    "What are the key dates and deadlines?",
    "List the most important numbers mentioned",
]
SUGGESTION_COUNT = 3
SUGGESTION_EXCERPT_CHARS = 2500

SUGGESTION_PROMPT = """You help people explore documents they uploaded. Based on the excerpts below, write the {count} questions a reader would most likely want answered from these documents.
Rules: each question must be answerable from the documents, specific to their content (mention concrete topics, never generic ones like "summarize this"), and under 12 words.
Return only a JSON array of {count} strings.

Document excerpts:
{excerpt}"""

ANSWER_PROMPT = """You answer questions about the user's documents using only the numbered passages below.
- Be direct and well structured. Use short paragraphs, and bullet lists or bold text where they help.
- After each sentence that uses a passage, cite it with its number in square brackets, like [1] or [2][4].
- If the passages don't contain the answer, say you couldn't find it in the documents. Never invent facts.

Earlier conversation (for context only):
{history}

Passages:
{context}

Question: {question}
Answer:"""


class UploadError(ValueError):
    """A problem with the uploaded files that the user can fix."""


def validate_files(files):
    """files: list of (filename, bytes)."""
    if not files:
        raise UploadError("Add at least one PDF.")
    if len(files) > MAX_FILES:
        raise UploadError(f"You can upload up to {MAX_FILES} PDFs at a time.")
    total_mb = sum(len(data) for _, data in files) / (1024 * 1024)
    if total_mb > MAX_TOTAL_MB:
        raise UploadError(f"These files add up to more than {MAX_TOTAL_MB:g} MB, which is the limit.")
    for name, data in files:
        if b"%PDF" not in data[:1024]:
            raise UploadError(f"“{name}” isn't a PDF.")


def extract_pages(files):
    """Returns (pages, docs): page texts with their source, and a summary per file."""
    pages, docs = [], []
    total_pages = total_chars = 0
    for name, data in files:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise UploadError(f"“{name}” is password protected.")
            page_count = len(reader.pages)
        except UploadError:
            raise
        except Exception:
            raise UploadError(f"“{name}” couldn't be opened. Is it a valid PDF?")
        total_pages += page_count
        if total_pages > MAX_PAGES:
            raise UploadError(f"These PDFs have more than {MAX_PAGES} pages in total.")
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            total_chars += len(text)
            if total_chars > MAX_CHARS:
                raise UploadError(f"These PDFs contain more than {MAX_CHARS:,} characters of text.")
            if text.strip():
                pages.append({"doc": name, "page": number, "text": text})
        docs.append({"name": name, "pages": page_count})
    if not pages:
        raise UploadError("No text could be read from these PDFs. Scanned documents without a text layer aren't supported.")
    return pages, docs


def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.create_documents(
        [p["text"] for p in pages],
        metadatas=[{"doc": p["doc"], "page": p["page"]} for p in pages],
    )


def build_store(chunks):
    return FAISS.from_documents(chunks, GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL))


@lru_cache(maxsize=4)
def text_chain(model):
    return ChatGoogleGenerativeAI(model=model) | StrOutputParser()


def sample_excerpt(text):
    # Beginning, middle and end, so long documents aren't judged by their cover page alone
    if len(text) <= SUGGESTION_EXCERPT_CHARS * 3:
        return text
    middle = len(text) // 2 - SUGGESTION_EXCERPT_CHARS // 2
    return "\n...\n".join([
        text[:SUGGESTION_EXCERPT_CHARS],
        text[middle:middle + SUGGESTION_EXCERPT_CHARS],
        text[-SUGGESTION_EXCERPT_CHARS:],
    ])


def parse_suggestions(raw):
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    questions = [q.strip() for q in items if isinstance(q, str) and q.strip()]
    return questions[:SUGGESTION_COUNT]


def generate_suggestions(pages):
    # Suggestions are a nice-to-have: any failure falls back to generic questions
    text = "\n".join(p["text"] for p in pages)
    prompt = SUGGESTION_PROMPT.format(count=SUGGESTION_COUNT, excerpt=sample_excerpt(text))
    try:
        try:
            raw = text_chain(SUGGESTION_MODEL).invoke(prompt)
        except Exception as e:
            if not is_rate_limit(e) or not FALLBACK_CHAT_MODEL or FALLBACK_CHAT_MODEL == SUGGESTION_MODEL:
                raise
            raw = text_chain(FALLBACK_CHAT_MODEL).invoke(prompt)
        questions = parse_suggestions(raw)
    except Exception as e:
        print(f"suggestions failed: {e!r}", flush=True)
        questions = []
    return questions if len(questions) == SUGGESTION_COUNT else FALLBACK_QUESTIONS


def retrieve(store, question, history):
    # Follow-ups like "and for interns?" need the previous question to find the right passages
    previous = [m["content"] for m in history if m["role"] == "user"][-1:]
    matches = store.similarity_search("\n".join(previous + [question]), k=TOP_K)
    sources = []
    for number, doc in enumerate(matches, start=1):
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
        sources.append({"id": number, "doc": doc.metadata["doc"], "page": doc.metadata["page"], "snippet": snippet})
    return matches, sources


def build_prompt(question, history, matches):
    context = "\n\n".join(
        f"[{n}] ({doc.metadata['doc']}, page {doc.metadata['page']})\n{doc.page_content}"
        for n, doc in enumerate(matches, start=1)
    )
    recent = history[-HISTORY_MESSAGES:]
    history_text = "\n".join(f"{m['role'].title()}: {m['content']}" for m in recent) or "(none)"
    return ANSWER_PROMPT.format(history=history_text, context=context, question=question)


def stream_answer(store, question, history):
    """Yields ("sources", list) once, then ("delta", text) chunks."""
    matches, sources = retrieve(store, question, history)
    yield "sources", sources
    prompt = build_prompt(question, history, matches)
    streamed = False
    try:
        for part in text_chain(CHAT_MODEL).stream(prompt):
            if part:
                streamed = True
                yield "delta", part
    except Exception as e:
        # Switching models mid-answer would repeat text, so only fall back before any is sent
        if streamed or not is_rate_limit(e) or not FALLBACK_CHAT_MODEL or FALLBACK_CHAT_MODEL == CHAT_MODEL:
            raise
        print(f"{CHAT_MODEL} rate limited, answering with {FALLBACK_CHAT_MODEL}", flush=True)
        for part in text_chain(FALLBACK_CHAT_MODEL).stream(prompt):
            if part:
                yield "delta", part


def is_rate_limit(error):
    text = f"{type(error).__name__} {error}"
    return "RESOURCE_EXHAUSTED" in text or "RateLimit" in text or "429" in text


def friendly_error(error):
    text = f"{type(error).__name__} {error}"
    if is_rate_limit(error):
        return "The free AI quota is used up for now. Please try again in a little while."
    if "API_KEY" in text or "UNAUTHENTICATED" in text or "PERMISSION_DENIED" in text:
        return "The AI service isn't configured correctly. Please try again later."
    return "Something went wrong while talking to the AI. Please try again."
