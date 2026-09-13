import streamlit as st
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import gc
import ctypes
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

SUGGESTED_QUESTIONS = [
    "Summarize these documents in 5 bullet points",
    "What are the key dates and deadlines?",
    "List the most important numbers mentioned",
]

CSS = """
<style>
#MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 860px;}
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);
    border-radius: 18px; padding: 28px 32px; color: white; margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
}
.hero h1 {color: white; font-size: 2rem; margin: 0 0 6px 0; padding: 0;}
.hero p {color: rgba(255,255,255,0.88); margin: 0; font-size: 1.02rem;}
.steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 8px 0 24px 0;}
.step {
    background: white; border: 1px solid #E5E7EB; border-radius: 14px; padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.step .num {
    display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px;
    border-radius: 50%; background: #EEF2FF; color: #4F46E5; font-weight: 700; margin-bottom: 8px;
}
.step h4 {margin: 0 0 4px 0; font-size: 1rem; color: #111827;}
.step p {margin: 0; color: #6B7280; font-size: 0.9rem;}
.doc-chip {
    background: #EEF2FF; color: #3730A3; border-radius: 10px; padding: 8px 10px;
    margin-bottom: 6px; font-size: 0.88rem; overflow-wrap: anywhere;
}
.doc-chip small {color: #6366F1;}
.brand {font-size: 1.35rem; font-weight: 800; color: #4F46E5; margin-bottom: 0.2rem;}
.muted {color: #6B7280; font-size: 0.85rem;}
[data-testid="stChatMessage"] {
    background: white; border: 1px solid #E5E7EB; border-radius: 14px; padding: 12px 16px;
}
@media (max-width: 640px) {
    .steps {grid-template-columns: 1fr;}
    .hero {padding: 22px;}
    .hero h1 {font-size: 1.6rem;}
}
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


def set_session_store(store, docs):
    registry = get_registry()
    with registry["lock"]:
        registry["stores"][current_session_id()] = {"store": store, "docs": docs, "last_used": time.time()}


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
    set_session_store(store, doc_info)
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


def main():
    st.set_page_config(page_title="ChatPDF", page_icon="📄", layout="centered")
    start_janitor()
    st.markdown(CSS, unsafe_allow_html=True)

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("uploader_key", 0)

    entry = get_session_entry()
    render_sidebar(entry)
    entry = get_session_entry()

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

    question = None
    if not st.session_state.messages:
        st.markdown('<div class="muted">Try one of these:</div>', unsafe_allow_html=True)
        cols = st.columns(len(SUGGESTED_QUESTIONS))
        for col, suggestion in zip(cols, SUGGESTED_QUESTIONS):
            if col.button(suggestion, use_container_width=True):
                question = suggestion

    typed = st.chat_input("Ask a question about your documents…")
    question = typed or question
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
    if typed is None:
        st.rerun()


if __name__ == "__main__":
    main()
