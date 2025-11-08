from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os

# Setup directories
os.makedirs("data/vector_store", exist_ok=True)

# Create embeddings
def create_embeddings(chunks):
    """ This function creates embeddings for text chunks using SentenceTransformer.
    Returns the model and the embeddings list.

        Params:
            chunks (int): represents the number of chunks from the original text

        Returns:
            model
            embeddings
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = [model.encode(chunk) for chunk in chunks]
    print("Embeddings created!")
    return model, embeddings

# Initialize Chroma vector store
def get_or_create_collection(chunks, embeddings):
    """
    Initialize or retrieve a Chroma collection with embeddings.
    """
    client = chromadb.Client(
        chromadb.config.Settings(
            persist_directory="data/vector_store",
            anonymized_telemetry=False
        )
    )

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="aws_rag_collection",
        embedding_function=embedding_function
    )

    # Add documents if collection is empty
    if len(collection.get()["ids"]) == 0:
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(chunks))]
        )
        print("Documents added to vector store.")
    else:
        print("Vector store already populated.")

    return collection, embedding_function


if __name__ == "__main__":
    pass
