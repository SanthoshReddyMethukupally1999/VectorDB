# vector_stores/faiss_store.py

from langchain.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def create_faiss_store(documents):
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(documents, embedding=embeddings)
    return vectordb
