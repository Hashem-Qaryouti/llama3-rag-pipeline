# RAG PDF Question-Answering with LLaMA

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG) pipeline** for PDF documents using:

- **LLaMA 3.2** (local GGUF model)
- **SentenceTransformers** for embeddings
- **Chroma** vector database
- **Streamlit** for an interactive web interface
- **PyPDF2** for PDF reading

Users can ask questions about a PDF, and the system retrieves relevant content and generates answers.

---

## Features

- Load and split PDF documents into smaller chunks
- Create embeddings with `SentenceTransformer`
- Store embeddings in a persistent **Chroma vector database**
- Query PDF content interactively using **LLaMA**
- Streamlit app for live Q&A with multiple questions
- Modular code for easy reuse

---

## Requirements

- Python 3.10+
- Libraries (can be installed via `requirements.txt`):

```bash
pip install -r requirements.txt

---

## Project Structure
.
├── data/
│   └── vector_store/        # Vector DB storage (ignored by git)
├── models/
│   └── Dolphin3.0-Llama3.2-1B-Q4_K_M.gguf   # Local LLaMA model (ignored by git)
├── src/
│   ├── ingest_pdf.py        # Load PDF and extract text
│   ├── split_text.py        # Split text into chunks
│   ├── create_vectorstore.py # Create embeddings & Chroma vector store
│   ├── RAG_pipeline.py      # Main functions: ask_question, load_llama_model
│   └── app.py               # Streamlit app
├── requirements.txt
└── README.md
