from dotenv import load_dotenv

import os
import re
import logging
import unicodedata
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

embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
db: Optional[Chroma] = None
llm: Optional[ChatGoogleGenerativeAI] = None

def normalize_text(text: str) -> str:
    try:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = text.replace(' .', '.').replace(' ,', ',')
        return text.strip()
    except Exception:
        logger.exception("Erro ao normalizar texto")
        return text

def index_pdf(pdf_path: str, embeddings_instance: GoogleGenerativeAIEmbeddings) -> bool:
    global db
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
        db = Chroma.from_documents(
            split_texts,
            embeddings_instance,
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

    if db is None or llm is None:
        logger.error("Serviços RAG (DB/LLM) não inicializados. Encerrando.")
        return "Erro interno: Serviço de conhecimento não inicializado."

    try:
        retriever = db.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        logger.info("Recuperados %d chunks relevantes", len(docs))
    except Exception:
        logger.exception("Erro ao recuperar documentos")
        return "Erro ao recuperar documentos dos procedimentos internos."


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
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        logger.info("Embeddings e LLM inicializados.")
    except Exception:
        logger.critical("Falha ao inicializar Embeddings ou LLM. Verifique as credenciais.")
        exit(1)

    chroma_db_dir = "./chroma_db"
    if os.path.isdir(chroma_db_dir):
        try:
            db = Chroma(
                persist_directory=chroma_db_dir,
                collection_name="internal_procedures",
                embedding_function=embeddings
            )
            logger.info("Base Chroma carregada do disco com sucesso.")
            ok = True
        except Exception:
            logger.warning("Erro ao carregar base Chroma. Tentando reindexar...")
            ok = index_pdf("procedures.pdf", embeddings)
    else:
        ok = index_pdf("procedures.pdf", embeddings)

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
