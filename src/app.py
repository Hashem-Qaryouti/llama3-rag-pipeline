import streamlit as st
from RAG_pipeline import load_and_split_pdf, setup_vectorstore, load_llama_model, ask_question

# Streamlit page config
st.set_page_config(page_title="RAG PDF Q&A", page_icon="📄")

st.title("📄 RAG PDF Question-Answering")
st.write("Ask questions about your PDF and get answers using LLaMA + RAG!")


# Load PDF and setup vector store (run once)
@st.cache_resource
def initialize_pipeline():
    chunks = load_and_split_pdf()
    collection, embedding_function = setup_vectorstore(chunks)
    llm = load_llama_model()
    return collection, embedding_function, llm

collection, embedding_function, llm = initialize_pipeline()


# User input
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question!")
    else:
        with st.spinner("Generating answer..."):
            answer = ask_question(question, collection, llm, embedding_function)
        st.markdown("**Answer:**")
        st.write(answer)
