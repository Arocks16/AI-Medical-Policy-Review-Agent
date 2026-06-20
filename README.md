<<<<<<< HEAD
# 🩺 MedVault — AI Medical Claim Assistant

**See your claim through the lens of your policy.**

MedVault is an AI-powered medical claim processing assistant. Upload your medical documents (Prescription, Lab Report, Bill), let Gemini OCR extract the data, and then chat with an AI agent that cross-references your claim against your insurance policy using RAG.


---

## Features

- **📄 Document Scanning** — Upload PDF/JPEG documents; text PDFs extracted via PyMuPDF, scanned PDFs/JPEGs sent to Gemini Vision for OCR
- **🧠 AI Chat Agent** — LangGraph-powered conversational agent with session memory (InMemorySaver)
- **📖 RAG Policy Lookup** — FAISS vector search over your insurance policy document to retrieve relevant clauses
- **🔒 Session Isolation** — Everything forgotten when you close the browser tab

---

## Architecture

```
frontend.py  ──►  ocr_utils.py  ──► Gemini Vision (OCR)
     │
     ├──► retrieval.py ──► FAISS index ──► Policy clauses
     │
     └──► backend.py  ──► Gemini 2.5 Flash (Chat)
```

| File | Role |
|---|---|
| `frontend.py` | Streamlit UI (2-tab: Process Documents + Chat) |
| `backend.py` | LangGraph chat agent with InMemorySaver checkpointing |
| `ocr_utils.py` | OCR pipeline — text/scanned PDFs & JPEGs → JSON |
| `retrieval.py` | FAISS vector store — lazy load + similarity search |
| `rag.py` | (Optional) Script to rebuild the FAISS index from `policy.txt` |

---

## Setup

```bash
# Install dependencies (using uv)
uv sync

# Or using pip
pip install -r requirements.txt

# Build the FAISS index (one time)
python rag.py

# Run the app
streamlit run frontend.py
```

> **Note:** The `faiss_index/` folder is pre-built. Run `python rag.py` only if you modify `policy.txt`.

---

## Environment

- **LLM:** Google Gemini 2.5 Flash (`gemini-2.5-flash`)
- **Embeddings:** Google Gemini Embedding (`models/gemini-embedding-001`)
- **Vector Store:** FAISS (local, CPU)
- **OCR:** Gemini Vision (for scanned/JPEG docs) + PyMuPDF (for text PDFs)
- **UI:** Streamlit
- **Agent Framework:** LangGraph with InMemorySaver

---

## Usage

1. **Process Documents tab** — Upload Prescription, Bills, and Lab Reports (PDF or JPEG, ≤500 KB each)
2. Click **Submit** — OCR extracts Patient name, Diagnosis, Bill amount, admission/discharge dates
3. Switch to **Chat tab** — Ask questions like *"Is my hospitalization covered?"* or *"Will my diabetes claim be approved?"*
4. The AI retrieves relevant policy clauses from the FAISS index and answers based on both your claim data and the policy

---

## Project Structure

```
├── frontend.py          # Streamlit UI
├── backend.py           # LangGraph chat agent
├── ocr_utils.py         # OCR processing
├── retrieval.py         # FAISS + embeddings (query-time)
├── rag.py               # Index building script
├── policy.txt           # Insurance policy text
├── faiss_index/         # Pre-built FAISS vector index
├── .env                 # API key
├── pyproject.toml       # Dependencies (uv)
└── requirements.txt     # Dependencies (pip)
```

---

MedVault is an AI-powered medical claim processing assistant. Upload your medical documents (Prescription, Lab Report, Bill), let Gemini OCR extract the data, and then chat with an AI agent that cross-references your claim against your insurance policy using RAG.
