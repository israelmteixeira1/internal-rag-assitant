import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import Optional

from app.utils.text_utils import normalize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("index_pdf")

def index_pdf(pdf_path: str, embeddings_instance: GoogleGenerativeAIEmbeddings) -> Optional[Chroma]:
    """Indexa um PDF e retorna a instância do Chroma DB."""
    logger.info("Iniciando indexação do PDF: %s", pdf_path)

    if not os.path.exists(pdf_path):
        logger.error("Arquivo PDF não encontrado: %s", pdf_path)
        return None

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info("PDF carregado com %d páginas", len(documents))
    except Exception:
        logger.exception("Erro ao carregar o PDF")
        return None

    try:
        for doc in documents:
            doc.page_content = normalize_text(doc.page_content)
    except Exception:
        logger.exception("Erro ao normalizar os documentos")
        return None

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_texts = text_splitter.split_documents(documents)
        logger.info("Documentos divididos em %d blocos", len(split_texts))
    except Exception:
        logger.exception("Erro durante a divisão dos blocos")
        return None

    if not split_texts:
        logger.critical("Nenhum bloco gerado a partir do PDF")
        return None

    try:
        chroma_db = Chroma.from_documents(
            split_texts,
            embeddings_instance,
            persist_directory="./chroma_db",
            collection_name="internal_procedures"
        )
        logger.info("Indexação concluída com sucesso")
        return chroma_db
    except Exception:
        logger.exception("Erro ao criar embeddings ou persistir no Chroma")
        return None