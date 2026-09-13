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


# One FAISS index per browser session, kept outside st.session_state so a
# background thread can flush it once the session is gone.
_stores = {}
_stores_lock = threading.Lock()


def _release_memory():
    gc.collect()
    try:
        # Hand freed heap pages back to the OS so the container's memory actually drops
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def flush_expired_stores():
    now = time.time()
    rt = runtime.get_instance() if runtime.exists() else None
    with _stores_lock:
        expired = [
            session_id for session_id, entry in _stores.items()
            if now - entry["last_used"] > IDLE_TIMEOUT_SECONDS
            or (rt is not None and not rt.is_active_session(session_id))
        ]
        for session_id in expired:
            del _stores[session_id]
    if expired:
        _release_memory()
    return expired


@st.cache_resource
def start_janitor():
    def loop():
        while True:
            time.sleep(JANITOR_INTERVAL_SECONDS)
            flush_expired_stores()

    threading.Thread(target=loop, name="faiss-janitor", daemon=True).start()


def current_session_id():
    return get_script_run_ctx().session_id


def get_session_store():
    with _stores_lock:
        entry = _stores.get(current_session_id())
        if entry is None:
            return None
        entry["last_used"] = time.time()
        return entry["store"]


def set_session_store(store):
    with _stores_lock:
        _stores[current_session_id()] = {"store": store, "last_used": time.time()}


def clear_session_store():
    with _stores_lock:
        removed = _stores.pop(current_session_id(), None)
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
    text=""
    pages = 0
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        pages += len(pdf_reader.pages)
        if pages > MAX_PAGES:
            raise ValueError(f"The PDFs have more than {MAX_PAGES} pages in total.")
        for page in pdf_reader.pages:
            text+= page.extract_text() or ""
            if len(text) > MAX_CHARS:
                raise ValueError(f"The PDFs contain more than {MAX_CHARS:,} characters of text.")
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = EMBEDDING_MODEL)
    set_session_store(FAISS.from_texts(text_chunks, embedding=embeddings))


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model=CHAT_MODEL,
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = prompt | model | StrOutputParser()

    return chain



def user_input(user_question):
    vector_store = get_session_store()
    if vector_store is None:
        st.warning("Upload your PDF files and click Submit & Process first.")
        return

    docs = vector_store.similarity_search(user_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = get_conversational_chain()
    response = chain.invoke({"context": context, "question": user_question})

    st.write("Reply: ", response)




def main():
    st.set_page_config("Chat PDF")
    start_janitor()
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        st.caption(
            f"Up to {MAX_FILES} PDFs, {MAX_TOTAL_MB:g} MB and {MAX_PAGES} pages in total. "
            f"Your documents are deleted when you leave or after {IDLE_TIMEOUT_SECONDS // 60} minutes of inactivity."
        )
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            error = validate_uploads(pdf_docs) if pdf_docs else "Please upload at least one PDF."
            if error:
                st.warning(error)
            else:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                    except ValueError as e:
                        st.warning(str(e))
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
        if get_session_store() is not None and st.button("Clear my documents"):
            clear_session_store()
            st.success("Your documents were deleted.")



if __name__ == "__main__":
    main()
