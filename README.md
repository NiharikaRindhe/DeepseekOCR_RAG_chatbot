# DeepSeek OCR & Nomic Local RAG (Ollama)

A simple Streamlit app for building a **local RAG (Retrieval-Augmented Generation)** system over PDF documents using:

- **DeepSeek OCR / Vision model** for text extraction (optional)
- **Nomic embeddings** for vector search
- **Local chat model** (e.g. DeepSeek-R1 via Ollama or other backends through `rag_engine`)
- **ChromaDB** (via `rag_engine`) as the vector store

The app lets you upload a PDF, index it into a local knowledge base, and then **chat with the document**.

---

## Features

- 📄 Upload and process **PDF documents**
- 🔎 Choose between:
  - **Built-in PDF text parsing**, or  
  - **Vision/OCR-based extraction** (for scanned PDFs) using a configurable model
- 🧠 Embed extracted text using a configurable **embedding model** (default: `nomic-embed-text`)
- 💬 Ask questions in a chat interface powered by a configurable **chat model** (default: `deepseek-r1`)
- 🧹 One-click **“Clear Knowledge Base”** to reset ChromaDB collection
- 📜 Expandable preview of extracted text

---

## Project Structure

Minimal layout (you can adapt as needed):

```text
.
├── app.py              # Streamlit app (the code you shared)
├── rag_engine.py       # Your RAG logic: PDF processing, embeddings, querying, Chroma client
├── requirements.txt    # Python dependencies (optional but recommended)
└── README.md           # This file
