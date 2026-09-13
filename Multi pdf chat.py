import streamlit as st
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import gc
import ctypes
import json
import threading
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.6-flash")

# Upload limits keep memory use predictable on a small server.
# The per-file limit is also enforced by Streamlit itself (server.maxUploadSize).
MAX_FILES = int(os.getenv("MAX_FILES", "3"))
MAX_TOTAL_MB = float(os.getenv("MAX_TOTAL_MB", "10"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "100"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "500000"))

# Indexes are dropped when their session ends, or after this long without use.
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_MINUTES", "15")) * 60
JANITOR_INTERVAL_SECONDS = 60

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

CSS = """
<style>
:root {
    --bg: #F7F7FB; --surface: #FFFFFF; --surface-2: #F3F4F6; --border: #E5E7EB;
    --text: #1F2937; --heading: #111827; --muted: #6B7280;
    --chip-bg: #EEF2FF; --chip-text: #3730A3; --chip-sub: #6366F1; --num-bg: #EEF2FF;
    --accent: #4F46E5;
}
#MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 860px;}
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);
    border-radius: 18px; padding: 28px 32px; color: white; margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
}
.hero h1 {color: white !important; font-size: 2rem; margin: 0 0 6px 0; padding: 0;}
.hero p {color: rgba(255,255,255,0.88) !important; margin: 0; font-size: 1.02rem;}
.steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 8px 0 24px 0;}
.step {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.step .num {
    display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px;
    border-radius: 50%; background: var(--num-bg); color: var(--accent); font-weight: 700; margin-bottom: 8px;
}
.step h4 {margin: 0 0 4px 0; font-size: 1rem; color: var(--heading) !important;}
.step p {margin: 0; color: var(--muted) !important; font-size: 0.9rem;}
.doc-chip {
    background: var(--chip-bg); color: var(--chip-text); border-radius: 10px; padding: 8px 10px;
    margin-bottom: 6px; font-size: 0.88rem; overflow-wrap: anywhere;
}
.doc-chip small {color: var(--chip-sub);}
.brand {font-size: 1.35rem; font-weight: 800; color: var(--accent); margin-bottom: 0.2rem;}
.muted {color: var(--muted) !important; font-size: 0.85rem;}
[data-testid="stChatMessage"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 12px 16px;
}
.framing {display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 0.85rem; margin: 4px 0 8px 0;}
.framing .sparkle {display: inline-block; animation: sparkle 1.4s ease-in-out infinite;}
.framing .dots::after {content: ""; animation: dots 1.5s steps(4, end) infinite;}
.skeleton {
    height: 38px; border-radius: 8px; margin-bottom: 10px; border: 1px solid var(--border);
    background: linear-gradient(90deg, var(--surface-2) 0%, var(--surface) 40%, var(--chip-bg) 50%, var(--surface) 60%, var(--surface-2) 100%);
    background-size: 300% 100%; animation: shimmer 1.6s ease-in-out infinite;
}
.skeleton:nth-child(2) {width: 88%; animation-delay: 0.15s;}
.skeleton:nth-child(3) {width: 94%; animation-delay: 0.3s;}
@keyframes shimmer {0% {background-position: 100% 0;} 100% {background-position: 0 0;}}
@keyframes sparkle {0%, 100% {transform: scale(1) rotate(0deg); opacity: 0.7;} 50% {transform: scale(1.3) rotate(20deg); opacity: 1;}}
@keyframes dots {0% {content: "";} 25% {content: ".";} 50% {content: "..";} 75% {content: "...";}}
@media (prefers-reduced-motion: reduce) {.skeleton, .framing .sparkle, .framing .dots::after {animation: none;}}
@media (max-width: 640px) {
    .steps {grid-template-columns: 1fr;}
    .hero {padding: 22px;}
    .hero h1 {font-size: 1.6rem;}
}
</style>
"""

# Streamlit's theme is set per server, so dark mode is applied per session by
# overriding the palette and the widgets that read colours from the base theme.
DARK_CSS = """
<style>
:root {
    --bg: #0B1020; --surface: #151B2E; --surface-2: #1D2439; --border: #2A3350;
    --text: #E5E7EB; --heading: #F9FAFB; --muted: #9CA3AF;
    --chip-bg: #1E2350; --chip-text: #C7D2FE; --chip-sub: #A5B4FC; --num-bg: #262B5C;
    --accent: #818CF8;
    color-scheme: dark;
}
[data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {
    background-color: var(--bg) !important;
}
[data-testid="stHeader"] {background: transparent !important;}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: var(--surface) !important; border-right: 1px solid var(--border);
}
[data-testid="stApp"] p, [data-testid="stApp"] li, [data-testid="stApp"] label,
[data-testid="stApp"] span, [data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] {
    color: var(--text);
}
[data-testid="stApp"] h1, [data-testid="stApp"] h2, [data-testid="stApp"] h3,
[data-testid="stApp"] strong {color: var(--heading);}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: var(--muted) !important;}
[data-testid="stSidebarCollapseButton"] svg, [data-testid="stExpandSidebarButton"] svg {fill: var(--text);}
[data-testid="stFileUploaderDropzone"] {background-color: var(--surface-2) !important; color: var(--text);}
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small {color: var(--muted) !important;}
/* Uploaded-file rows use hard-coded light backgrounds */
[data-testid="stFileUploader"] div:not([data-testid="stFileUploaderDropzone"]) {background-color: transparent !important;}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] div {color: var(--text);}
[data-testid="stBaseButton-secondary"] {
    background-color: var(--surface-2) !important; color: var(--text) !important; border-color: var(--border) !important;
}
[data-testid="stBaseButton-secondary"]:hover {border-color: var(--accent) !important; color: var(--heading) !important;}
[data-testid="stBaseButton-primary"] {background-color: #6366F1 !important; border-color: #6366F1 !important;}
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {
    background-color: var(--surface) !important; border-color: var(--border) !important;
}
[data-testid="stChatInputTextArea"] {background-color: var(--surface) !important; color: var(--text) !important;}
[data-testid="stChatInputTextArea"]::placeholder {color: var(--muted) !important;}
[data-testid="stAlert"] > div {background-color: var(--surface-2) !important;}
[data-testid^="stChatMessageAvatar"] {background-color: var(--surface-2) !important; color: var(--text);}
hr {border-color: var(--border) !important;}
</style>
"""


# One FAISS index per browser session. Streamlit re-executes this script on every
# interaction, so the registry has to live in st.cache_resource (created once per
# server process) rather than a module-level global, which would be reset each time.
@st.cache_resource
def get_registry():
    return {"stores": {}, "lock": threading.Lock()}


def _release_memory():
    gc.collect()
    try:
        # Hand freed heap pages back to the OS so the container's memory actually drops
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def flush_expired_stores(registry):
    now = time.time()
    rt = runtime.get_instance() if runtime.exists() else None
    with registry["lock"]:
        expired = [
            session_id for session_id, entry in registry["stores"].items()
            if now - entry["last_used"] > IDLE_TIMEOUT_SECONDS
            or (rt is not None and not rt.is_active_session(session_id))
        ]
        for session_id in expired:
            del registry["stores"][session_id]
    if expired:
        _release_memory()
    return expired


@st.cache_resource
def start_janitor():
    registry = get_registry()

    def loop():
        while True:
            time.sleep(JANITOR_INTERVAL_SECONDS)
            flush_expired_stores(registry)

    threading.Thread(target=loop, name="faiss-janitor", daemon=True).start()


def current_session_id():
    return get_script_run_ctx().session_id


def get_session_entry():
    registry = get_registry()
    with registry["lock"]:
        entry = registry["stores"].get(current_session_id())
        if entry is not None:
            entry["last_used"] = time.time()
        return entry


def set_session_store(store, docs, suggestions=None):
    registry = get_registry()
    entry = {"store": store, "docs": docs, "suggestions": suggestions, "last_used": time.time()}
    with registry["lock"]:
        registry["stores"][current_session_id()] = entry
    return entry


def start_suggestion_worker(entry, text):
    # Runs outside the script thread so the chat stays usable while Gemini thinks
    registry = get_registry()

    def work():
        suggestions = generate_suggestions(text)
        with registry["lock"]:
            # The entry may have been cleared or replaced meanwhile; writing to it is then harmless
            entry["suggestions"] = suggestions

    threading.Thread(target=work, name="suggestions", daemon=True).start()


def clear_session_store():
    registry = get_registry()
    with registry["lock"]:
        removed = registry["stores"].pop(current_session_id(), None)
    if removed:
        _release_memory()


def validate_uploads(pdf_docs):
    if len(pdf_docs) > MAX_FILES:
        return f"Please upload at most {MAX_FILES} PDFs at a time."
    total_mb = sum(pdf.size for pdf in pdf_docs) / (1024 * 1024)
    if total_mb > MAX_TOTAL_MB:
        return f"Total upload is {total_mb:.1f} MB; the limit is {MAX_TOTAL_MB:g} MB."
    return None


def get_pdf_text(pdf_docs):
    text = ""
    pages = 0
    doc_info = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pages += len(pdf_reader.pages)
        if pages > MAX_PAGES:
            raise ValueError(f"The PDFs have more than {MAX_PAGES} pages in total.")
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
            if len(text) > MAX_CHARS:
                raise ValueError(f"The PDFs contain more than {MAX_CHARS:,} characters of text.")
        doc_info.append({"name": pdf.name, "pages": len(pdf_reader.pages)})
    if not text.strip():
        raise ValueError("No text could be read from these PDFs. Scanned documents without a text layer aren't supported.")
    return text, doc_info


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def build_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_texts(text_chunks, embedding=embeddings)


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


def generate_suggestions(text):
    # Suggestions are a nice-to-have: any failure, including model setup, falls back to generic questions
    try:
        prompt = PromptTemplate.from_template(SUGGESTION_PROMPT)
        chain = prompt | ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.4) | StrOutputParser()
        questions = parse_suggestions(chain.invoke({"count": SUGGESTION_COUNT, "excerpt": sample_excerpt(text)}))
    except Exception as e:
        print(f"suggestions failed: {e!r}", flush=True)
        questions = []
    return questions if len(questions) == SUGGESTION_COUNT else FALLBACK_QUESTIONS


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt | model | StrOutputParser()


def answer_question(vector_store, user_question):
    docs = vector_store.similarity_search(user_question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return get_conversational_chain().invoke({"context": context, "question": user_question})


def render_sidebar(entry):
    with st.sidebar:
        st.markdown('<div class="brand">📄 ChatPDF</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Ask questions about your documents, powered by Gemini.</div>', unsafe_allow_html=True)
        st.toggle("🌙 Dark mode", key="dark_mode")
        st.divider()

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if st.button("Process documents", type="primary", use_container_width=True):
            error = validate_uploads(pdf_docs) if pdf_docs else "Please upload at least one PDF."
            if error:
                st.warning(error)
            else:
                process_uploads(pdf_docs)
                st.rerun()

        if entry is not None:
            st.markdown("**Ready to chat**")
            for doc in entry["docs"]:
                st.markdown(
                    f'<div class="doc-chip">📘 {doc["name"]}<br><small>{doc["pages"]} page(s)</small></div>',
                    unsafe_allow_html=True,
                )
            if st.button("Clear documents & chat", use_container_width=True):
                clear_session_store()
                st.session_state.messages = []
                st.session_state.uploader_key += 1
                st.rerun()

        st.divider()
        st.caption(
            f"Limits: {MAX_FILES} PDFs, {MAX_TOTAL_MB:g} MB, {MAX_PAGES} pages. "
            f"Documents are deleted when you leave or after {IDLE_TIMEOUT_SECONDS // 60} minutes idle."
        )


def process_uploads(pdf_docs):
    with st.spinner("Reading and indexing your documents…"):
        try:
            raw_text, doc_info = get_pdf_text(pdf_docs)
            store = build_vector_store(get_text_chunks(raw_text))
        except ValueError as e:
            st.session_state.flash = ("warning", str(e))
            return
        except Exception as e:
            st.session_state.flash = ("error", f"Couldn't index the documents: {type(e).__name__}. Please try again.")
            print(f"indexing failed: {e!r}", flush=True)
            return
    entry = set_session_store(store, doc_info)
    start_suggestion_worker(entry, raw_text)
    st.session_state.messages = []
    st.session_state.flash = ("success", f"Indexed {len(doc_info)} document(s). Ask away!")


def render_empty_state():
    st.markdown(
        """
        <div class="steps">
          <div class="step"><div class="num">1</div><h4>Upload</h4><p>Add up to 3 PDFs from the sidebar.</p></div>
          <div class="step"><div class="num">2</div><h4>Process</h4><p>Click <b>Process documents</b> to index them.</p></div>
          <div class="step"><div class="num">3</div><h4>Ask</h4><p>Chat with your documents below.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggestions(suggestions):
    st.markdown('<div class="muted">✨ Suggested for these documents:</div>', unsafe_allow_html=True)
    # One per row: AI-written questions are too long for side-by-side columns
    for i, suggestion in enumerate(suggestions):
        if st.button(f"💬 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
            st.session_state.picked_suggestion = suggestion
            st.rerun()


# Only this fragment re-runs while Gemini writes suggestions, so the chat input
# below stays usable and keeps whatever the user is typing.
@st.fragment(run_every=1)
def render_pending_suggestions():
    entry = get_session_entry()
    if entry is None or entry.get("suggestions") is not None:
        st.rerun(scope="app")
    st.markdown(
        '<div class="framing"><span class="sparkle">✨</span>'
        '<span>Framing questions for your documents<span class="dots"></span> '
        "You can start typing below anytime.</span></div>"
        '<div><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="ChatPDF", page_icon="📄", layout="centered")
    start_janitor()

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("dark_mode", False)

    entry = get_session_entry()
    render_sidebar(entry)
    entry = get_session_entry()

    # Injected after the toggle so a switch takes effect on the same run
    st.markdown(CSS, unsafe_allow_html=True)
    if st.session_state.dark_mode:
        st.markdown(DARK_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="hero"><h1>Chat with your PDFs</h1>'
        "<p>Upload documents, then ask anything. Answers come only from what's inside them.</p></div>",
        unsafe_allow_html=True,
    )

    flash = st.session_state.pop("flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])

    if entry is None:
        if st.session_state.messages:
            st.info("Your documents were cleared after inactivity. Upload them again to keep chatting.")
            st.session_state.messages = []
        render_empty_state()
        st.chat_input("Upload and process a PDF to start chatting", disabled=True)
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "📄"):
            st.markdown(message["content"])

    if not st.session_state.messages:
        if entry.get("suggestions") is None:
            render_pending_suggestions()
        else:
            render_suggestions(entry["suggestions"])

    typed = st.chat_input("Ask a question about your documents…")
    question = typed or st.session_state.pop("picked_suggestion", None)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Thinking…"):
            try:
                answer = answer_question(entry["store"], question)
            except Exception as e:
                print(f"answer failed: {e!r}", flush=True)
                answer = f"⚠️ Sorry, I couldn't get an answer right now ({type(e).__name__}). Please try again."
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    # Redraw from history so the suggestion buttons disappear after the first question
    st.rerun()


if __name__ == "__main__":
    main()
