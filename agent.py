import os
from dotenv import load_dotenv
from langchain.llms import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

# Placeholder: you can replace with Groq integration if available in LangChain
def run_qa_chain(vectordb, query):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    
    llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="mistral-saba-24b",
    temperature=1
      # or "mistral-7b"
    )

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
    )

    result = qa_chain.run(query)
    return result
