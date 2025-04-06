# vector_stores/pinecone_store.py

import os
import pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings

def create_pinecone_store(documents, index_name="vectordb-index"):
    # Load API key and environment
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    embeddings = HuggingFaceEmbeddings()

    vectordb = LangchainPinecone.from_documents(
        documents,
        embedding=embeddings,
        index_name=index_name
    )
    return vectordb
