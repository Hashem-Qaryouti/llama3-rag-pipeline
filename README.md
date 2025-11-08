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
