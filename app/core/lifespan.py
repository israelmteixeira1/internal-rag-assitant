from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
import os

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.services.indexing_service import index_pdf
from app.core import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("lifespan")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    
    Na inicialização (startup):
    1. Carrega variáveis de ambiente (.env)
    2. Inicializa embeddings do Google (text-embedding-004)
    3. Inicializa LLM do Google (gemini-2.0-flash)
    4. Carrega Chroma DB do disco OU indexa o PDF se necessário
    
    No shutdown:
    - Libera recursos (se necessário)
    """

    # 1. Carregar variáveis de ambiente
    load_dotenv()
    logger.info("Variáveis de ambiente carregadas")
    
    # 2. Inicializar Embeddings
    try:
        state.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        logger.info("Embeddings inicializados com sucesso")
    except Exception:
        logger.critical("Falha ao inicializar os Embeddings. Verifique suas credenciais.")
        raise
    
    # 3. Inicializar LLM
    try:
        state.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, request_timeout=20)
        logger.info("LLM inicializado com sucesso")
    except Exception:
        logger.critical("Falha ao inicializar o LLM. Verifique suas credenciais.")
        raise
    
    # 4. Carregar ou criar Chroma DB
    chroma_db_dir = "./chroma_db"
    if os.path.isdir(chroma_db_dir):
        try:
            state.db = Chroma(
                persist_directory=chroma_db_dir,
                collection_name="internal_procedures",
                embedding_function=state.embeddings
            )
            logger.info("Chroma DB carregado do disco com sucesso")
        except Exception:
            logger.warning("Erro ao carregar Chroma DB. Tentando reindexar...")
            state.db = index_pdf("procedures.pdf", state.embeddings)
    else:
        logger.info("Chroma DB não encontrado. Indexando PDF...")
        state.db = index_pdf("procedures.pdf", state.embeddings)
    
    if state.db is None:
        logger.critical("Falha crítica durante a indexação. A aplicação não pode iniciar.")
        raise RuntimeError("Falha ao inicializar Chroma DB")
    
    logger.info("API RAG pronta para receber requisições")
    
    yield
    
    logger.info("Encerrando API RAG")