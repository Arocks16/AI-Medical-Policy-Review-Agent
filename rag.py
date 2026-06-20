import os
import re
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE, ".env"))

POLICY_PATH = os.path.join(_BASE, "policy.txt")
INDEX_DIR = os.path.join(_BASE, "faiss_index")

_vector_store = None


def _ensure_index():
    global _vector_store
    if _vector_store is None and os.path.isdir(INDEX_DIR):
        api_key = os.getenv("llm_key") or os.getenv("GOOGLE_API_KEY")
        emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        _vector_store = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)


def get_context(query: str, k: int = 4) -> str:
    _ensure_index()
    if _vector_store is None:
        return ""
    docs = _vector_store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


def _clean_text(text: str) -> str:
    text = re.sub(
        r"United India Insurance Company Limited.*?UIN:.*?(?=\n|$)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"\d+\s*\|\s*P\s*a\s*g\s*e", "", text)
    text = re.sub(r"\(.*?Code.*?\)", "", text)
    text = re.sub(r"\d+\.\d+\s*\|", "", text)
    text = text.replace("\u25aa", "-").replace("\u2022", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [l.strip() for l in text.split("\n")]
    return "\n".join(l for l in lines if l)


def build_index():
    print("Loading policy text...")
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean = _clean_text(raw_text)
    print(f"Text loaded and cleaned ({len(clean)} chars)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = [Document(page_content=c) for c in splitter.split_text(clean)]
    print(f"Split into {len(chunks)} chunks")

    api_key = os.getenv("llm_key") or os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    for attempt in range(3):
        try:
            store = FAISS.from_documents(chunks, embeddings)
            os.makedirs(INDEX_DIR, exist_ok=True)
            store.save_local(INDEX_DIR)
            print(f"Index saved to {INDEX_DIR}/")
            return
        except GoogleGenerativeAIError as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" not in err_str:
                raise
            is_daily = "PerDay" in err_str
            if attempt == 2:
                msg = (
                    "Google Embedding API daily quota exhausted. "
                ) if is_daily else (
                    "Google Embedding API per-minute quota exhausted.\n"
                )
                raise RuntimeError(msg + (
                    "Get a fresh API key from https://aistudio.google.com/app/apikey "
                    "and update it in your .env file."
                ))
            m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
            wait = max(65, int(float(m.group(1)) + 10)) if m else 65
            print(f"Quota full. Waiting {wait}s before retry...")
            time.sleep(wait)


if __name__ == "__main__":
    build_index()
