"""
RAG Engine — Handles PDF/Document ingestion and conversational Q&A.

Design choices:
- HuggingFaceEmbeddings (all-mpnet-base-v2)
- Chroma persisted under Vector/pdf_documents/ (separate from invoices)
- Each uploaded file gets its own Chroma collection (session_id)
- ChatGroq (llama-3.3-70b-versatile) as the LLM
- LCEL chain with message-history for follow-up question support
"""

import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
# Separate from invoices collection which lives in Vector/invoices/
VECTOR_DIR = Path(__file__).parent.parent / "Vector" / "pdf_documents"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# ── Singletons ────────────────────────────────────────────────────────────────
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_llm: Optional[ChatGroq] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    return _embedding_model


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=4096,  # allow long, detailed responses
        )
    return _llm


# ── In-memory session state ───────────────────────────────────────────────────
# { session_id: {"history": [HumanMessage, AIMessage, ...], "filename": str} }
_sessions: dict[str, dict] = {}


# ── Document ingestion ────────────────────────────────────────────────────────
def load_document(file_path: str) -> list:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


def ingest_document(file_path: str, filename: str) -> str:
    """Load → chunk → embed → persist in Chroma. Returns session_id."""
    session_id = str(uuid.uuid4())
    docs = load_document(file_path)

    for doc in docs:
        doc.metadata["uploaded_filename"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        collection_name=session_id,
        persist_directory=str(VECTOR_DIR),
    )

    _sessions[session_id] = {"history": [], "filename": filename}
    return session_id


# ── Vector store helper ───────────────────────────────────────────────────────
def _get_retriever(session_id: str):
    vs = Chroma(
        collection_name=session_id,
        embedding_function=get_embedding_model(),
        persist_directory=str(VECTOR_DIR),
    )
    return vs.as_retriever(search_kwargs={"k": 8})  # fetch more chunks for richer context


# ── Prompt templates ──────────────────────────────────────────────────────────
_contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, "
     "reformulate the question to be standalone so it can be understood "
     "without the history. Do NOT answer. Just reformulate if needed, else return as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Expert research assistant. Answer ONLY using provided context. No outside info.\n\n"
     "- Use bullet points for clarity.\n"
     "- Be concise: provide short, direct answers. Ask 'Do you need more information?' and stop. Provide thorough details ONLY if the user says yes.\n"
     "- Quote context to support answers. Address all query aspects.\n"
     "- If the answer is missing, state what is available in the context instead.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ── RAG chain ─────────────────────────────────────────────────────────────────
def _run_rag(session_id: str, question: str) -> tuple[str, list]:
    """
    Runs the two-step RAG process:
      1. Rephrase follow-up questions into standalone queries.
      2. Retrieve relevant chunks and generate the answer.
    Returns (answer_str, source_docs).
    """
    llm = get_llm()
    retriever = _get_retriever(session_id)
    history = _sessions[session_id]["history"]

    # Step 1: Rephrase if there is history
    if history:
        rephrased = (
            _contextualize_prompt
            | llm
            | StrOutputParser()
        ).invoke({"input": question, "chat_history": history})
    else:
        rephrased = question

    # Step 2: Retrieve + Generate
    docs = retriever.invoke(rephrased)
    context = _format_docs(docs)

    answer = (
        _qa_prompt
        | llm
        | StrOutputParser()
    ).invoke({"input": question, "context": context, "chat_history": history})

    return answer, docs


# ── Public API ────────────────────────────────────────────────────────────────
def chat(session_id: str, question: str) -> dict:
    if session_id not in _sessions:
        # Try to restore from persisted Chroma collection
        try:
            Chroma(
                collection_name=session_id,
                embedding_function=get_embedding_model(),
                persist_directory=str(VECTOR_DIR),
            ).get()
            _sessions[session_id] = {"history": [], "filename": "unknown"}
        except Exception:
            raise ValueError(f"Unknown session: {session_id}")

    answer, docs = _run_rag(session_id, question)

    # Update history
    _sessions[session_id]["history"].append(HumanMessage(content=question))
    _sessions[session_id]["history"].append(AIMessage(content=answer))

    # Build citation strings
    sources = list({
        doc.metadata.get("uploaded_filename", doc.metadata.get("source", ""))
        + (f" (page {doc.metadata['page'] + 1})" if "page" in doc.metadata else "")
        for doc in docs
    })

    return {"answer": answer, "sources": sources}


def list_sessions() -> list[dict]:
    return [
        {"session_id": sid, "filename": data["filename"]}
        for sid, data in _sessions.items()
    ]


def clear_session_history(session_id: str):
    if session_id in _sessions:
        _sessions[session_id]["history"] = []
