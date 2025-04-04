# loaders/doc_loader.py

from langchain.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path

def load_document(file_path: str):
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()
