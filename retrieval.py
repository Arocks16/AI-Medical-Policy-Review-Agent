import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

api_key = os.environ.get("llm_key")
_vector_store = None
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")


def _ensure_index():
    global _vector_store
    if _vector_store is None and os.path.isdir(INDEX_DIR):
        emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        _vector_store = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)


def get_context(query: str, k: int = 4) -> str:
    _ensure_index()
    if _vector_store is None:
        return ""
    docs = _vector_store.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)
