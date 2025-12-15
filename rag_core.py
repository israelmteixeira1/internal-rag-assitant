from dotenv import load_dotenv
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def normalize_text(text:str)->str:
    text=re.sub(r'\s+',' ',text)
    text=text.replace(' .','.').replace(' ,',',')
    return text.strip()

def index_pdf(pdf_path:str):
    loader=PyPDFLoader(pdf_path)
    documents=loader.load()
    for doc in documents:
        doc.page_content=normalize_text(doc.page_content)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    split_texts=text_splitter.split_documents(documents)
    embeddings=GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    db=Chroma.from_documents(split_texts,embeddings,persist_directory="./chroma_db",collection_name="internal_procedures")

def query_rag(question:str)->str:
    embeddings=GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    db=Chroma(persist_directory="./chroma_db",collection_name="internal_procedures",embedding_function=embeddings)
    retriever=db.as_retriever(search_kwargs={"k":4})
    docs=retriever.invoke(question)
    print("\n=== DOCUMENTOS RECUPERADOS ===\n")
    for i,doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:500])
        print("\nMETADATA:",doc.metadata)
        print("\n-----------------------------\n")
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.2)
    prompt=ChatPromptTemplate.from_template("""
You are an internal support assistant.

Answer the question ONLY using the information provided in the context.
If the answer is not present in the context, say:
"Não encontrei essa informação nos procedimentos internos."

Context:
{context}

Question:
{question}
""")
    rag_chain=(
        {
            "context":retriever|format_docs,
            "question":RunnablePassthrough()
        }
        |prompt
        |llm
    )
    response=rag_chain.invoke(question)
    return response.content

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__=="__main__":
    index_pdf("procedures.pdf")
    pergunta_teste = input("\nDigite a sua pergunta sobre os procedimentos internos (ou digite 'sair' para terminar): \n> ")

    if pergunta_teste.lower() != 'sair':
        print("\nProcessando...")
        
        # Consulta RAG com a pergunta inserida pelo usuário
        resultado = query_rag(pergunta_teste)
        
        print("\n--- RESPOSTA RAG ---")
        print(resultado)
        print("--------------------\n")
    else:
        print("Programa encerrado.")
