# app.py

import streamlit as st
from loaders.doc_loader import load_document
from vector_stores.faiss_store import create_faiss_store
from vector_stores.chroma_store import create_chroma_store
from vector_stores.pinecone_store import create_pinecone_store
from agent import run_qa_chain

st.set_page_config(page_title="VectorDB Comparison", layout="wide")

st.title("🧠 VectorDB Agent Compare")
st.markdown("Upload a document, ask a question, and compare answers from FAISS, Chroma, and Pinecone.")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
query = st.text_input("Ask your question:")

if uploaded_file and query:
    with st.spinner("Reading document..."):
        with open(f"temp_input.{uploaded_file.name.split('.')[-1]}", "wb") as f:
            f.write(uploaded_file.read())
        docs = load_document(f.name)

    with st.spinner("Creating vector stores..."):
        faiss_db = create_faiss_store(docs)
        chroma_db = create_chroma_store(docs)
        pinecone_db = create_pinecone_store(docs)

    with st.spinner("Getting answers..."):
        faiss_answer = run_qa_chain(faiss_db, query)
        chroma_answer = run_qa_chain(chroma_db, query)
        pinecone_answer = run_qa_chain(pinecone_db, query)

    st.subheader("Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### FAISS")
        st.success(faiss_answer)

    with col2:
        st.markdown("### Chroma")
        st.info(chroma_answer)

    with col3:
        st.markdown("### Pinecone")
        st.warning(pinecone_answer)
