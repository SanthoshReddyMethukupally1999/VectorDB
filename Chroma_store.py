# vector_stores/chroma_store.py

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def create_chroma_store(documents, persist_directory="chroma_db"):
    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb
