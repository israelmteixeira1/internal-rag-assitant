from dotenv import load_dotenv

import os
import re
import logging
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

logger = logging.getLogger("rag_pipeline")
load_dotenv()

def normalize_text(text: str) -> str:
    try:
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(' .', '.').replace(' ,', ',')
        return text.strip()
    except Exception:
        logger.exception("Erro ao normalizar texto")
        return text

def index_pdf(pdf_path: str) -> bool:
    logger.info("Iniciando indexação do PDF: %s", pdf_path)

    if not os.path.exists(pdf_path):
        logger.error("Arquivo PDF não encontrado: %s", pdf_path)
        return False

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info("PDF carregado com %d páginas", len(documents))
    except Exception:
        logger.exception("Erro ao carregar o PDF")
        return False

    try:
        for doc in documents:
            doc.page_content = normalize_text(doc.page_content)
    except Exception:
        logger.exception("Erro ao normalizar documentos")
        return False

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_texts = text_splitter.split_documents(documents)
        logger.info("Documentos divididos em %d chunks", len(split_texts))
    except Exception:
        logger.exception("Erro ao realizar chunking")
        return False

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004"
        )

        Chroma.from_documents(
            split_texts,
            embeddings,
            persist_directory="./chroma_db",
            collection_name="internal_procedures"
        )

        logger.info("Indexação concluída com sucesso")
        return True
    except Exception:
        logger.exception("Erro ao criar embeddings ou persistir no Chroma")
        return False

def format_docs(docs: List[Document]) -> str:
    try:
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception:
        logger.exception("Erro ao formatar documentos")
        return ""

def query_rag(question: str) -> Optional[str]:
    logger.info("Recebida pergunta: %s", question)

    if not question or not question.strip():
        logger.warning("Pergunta vazia ou inválida")
        return "Pergunta vazia ou inválida."

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004"
        )

        db = Chroma(
            persist_directory="./chroma_db",
            collection_name="internal_procedures",
            embedding_function=embeddings
        )
    except Exception:
        logger.exception("Erro ao carregar base vetorial")
        return "Erro ao carregar a base de conhecimento interna."

    try:
        retriever = db.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        logger.info("Recuperados %d chunks relevantes", len(docs))
    except Exception:
        logger.exception("Erro ao recuperar documentos")
        return "Erro ao recuperar documentos dos procedimentos internos."

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2
        )
    except Exception:
        logger.exception("Erro ao inicializar LLM")
        return "Erro ao inicializar o modelo de linguagem."

    prompt = ChatPromptTemplate.from_template("""
You are an internal support assistant.

Answer the question ONLY using the information provided in the context.
If the answer is not present in the context, say:
"Não encontrei essa informação nos procedimentos internos."

Context:
{context}

Question:
{question}
""")

    try:
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        response = rag_chain.invoke(question)
        logger.info("Resposta gerada com sucesso")
        return response.content
    except Exception:
        logger.exception("Erro ao executar cadeia RAG")
        return "Ocorreu um erro ao gerar a resposta com base nos procedimentos internos."

if __name__ == "__main__":
    ok = index_pdf("procedures.pdf")
    if not ok:
        logger.critical("Falha crítica durante indexação. Encerrando aplicação.")
        print("Erro ao indexar procedimentos internos. Verifique os logs.")
        exit(1)

    try:
        pergunta_teste = input(
            "\nDigite a sua pergunta sobre os procedimentos internos "
            "(ou digite 'sair' para terminar): \n> "
        )
    except (KeyboardInterrupt, EOFError):
        logger.info("Programa encerrado pelo usuário (interrupt/EOF).")
        print("\nPrograma encerrado.")
        exit(0)

    if pergunta_teste.lower() != "sair":
        print("\nProcessando...")

        resultado = query_rag(pergunta_teste)
        print("\n--- RESPOSTA RAG ---")
        print(resultado if resultado is not None else "Erro inesperado.")
        print("--------------------\n")
    else:
        logger.info("Programa encerrado pelo usuário.")
        print("Programa encerrado.")
