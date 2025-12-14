from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

def index_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    db = Chroma.from_documents(split_texts, embeddings, persist_directory="./chroma_db", collection_name="internal_procedures")

    db.persist()

def query_rag(question: str) -> str:
    db = Chroma(persist_directory="./chroma_db", collection_name="internal_procedures")
    retriever = db.as_retriever(search_kwargs={"k": 4})

    qa = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2),
        chain_type="stuff",
        retriever=retriever
    )

    response = qa({"query": question})

    if response['result']:
        return response['result']
    else:
        return "I couldn't find this information in the internal procedures."

if __name__ == "__main__":
    # Create a small dummy PDF called "procedures.pdf"
    # containing at least 5 paragraphs simulating internal procedures
    index_pdf("procedures.pdf")
    
    # Run a test by calling:
    result = query_rag("What is the step-by-step process to restart the system?")
    
    # Print the result in the terminal.
    print(result)