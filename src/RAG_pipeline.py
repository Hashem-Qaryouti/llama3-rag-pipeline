from ingest_pdf import load_pdf
from split_text import split_text
from create_vectorstore import create_embeddings, get_or_create_collection
from chromadb.utils import embedding_functions
from llama_cpp import Llama
import os

# -------------------------------
# 1️⃣ Load PDF and split
# -------------------------------
def load_and_split_pdf():
    text = load_pdf()
    chunks = split_text(text)
    print(f"Total chunks: {len(chunks)}")
    return chunks

# -------------------------------
# 2️⃣ Load LLaMA model
# -------------------------------
def load_llama_model(model_path="models/Dolphin3.0-Llama3.2-1B-Q4_K_M.gguf"):
    return Llama(model_path=model_path)

# -------------------------------
# 3️⃣ Setup vector store
# -------------------------------
def setup_vectorstore(chunks):
    model, embeddings = create_embeddings(chunks)
    collection, embedding_function = get_or_create_collection(chunks, embeddings)
    return collection, embedding_function

# -------------------------------
# 4️⃣ RAG query function
# -------------------------------
def ask_question(question, collection, llm, embedding_function,
                 n_results=3, max_chunks=3, max_chunk_len=500, max_tokens=200):
    # Retrieve top chunks
    results = collection.query(query_texts=[question], n_results=n_results)
    chunks = results['documents'][0]

    if not chunks:
        return "No relevant context found."

    # Limit number of chunks to avoid exceeding LLaMA context window
    context = "\n\n".join(chunk[:max_chunk_len] for chunk in chunks[:max_chunks])

    # Build prompt
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    
    # Generate answer
    output = llm(prompt, max_tokens=max_tokens)
    return output['choices'][0]['text'].strip()

# -------------------------------
# 5️⃣ Example usage
# -------------------------------
if __name__ == "__main__":
    # Load chunks
    chunks = load_and_split_pdf()
    
    # Setup vector store
    collection, embedding_function = setup_vectorstore(chunks)
    
    # Load LLaMA model once
    llm = load_llama_model()
    
    # Ask question
    question = "What are the best practices for RAG applications?"
    answer = ask_question(question, collection, llm, embedding_function)
    
    # Print question + answer
    print(f"\nQuestion:\n{question}\n")
    print(f"Answer:\n{answer}")
