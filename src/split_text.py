import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest_pdf import load_pdf

def split_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits the text into smaller chunks for embeddings and retrieval.

    Params:
        chunk_size (int): number of characters per chunk
        chunk_overlap (int): number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    # Load the text
    text = load_pdf()
    print(f"Total characters in PDF: {len(text)}\n")

    # Split text into chunks
    chunks = split_text(text)
    print(f"Text split into {len(chunks)} chunks.\n")

    # Show previews of the first 3 chunks
    print("First 3 chunk previews:\n")
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ---\n{chunk}\n")
