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
    logger.info("Starting PDF indexing: %s", pdf_path)

    if not os.path.exists(pdf_path):
        logger.error("PDF file not found: %s", pdf_path)
        return None

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info("PDF loaded with %d pages", len(documents))
    except Exception:
        logger.exception("Error loading PDF")
        return None

    try:
        for doc in documents:
            doc.page_content = normalize_text(doc.page_content)
    except Exception:
        logger.exception("Error normalizing documents")
        return None

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_texts = text_splitter.split_documents(documents)
        logger.info("Documents split into %d chunks", len(split_texts))
    except Exception:
        logger.exception("Error during chunking")
        return None

    if not split_texts:
        logger.critical("No chunks generated from PDF")
        return None

    try:
        chroma_db = Chroma.from_documents(
            split_texts,
            embeddings_instance,
            persist_directory="./chroma_db",
            collection_name="internal_procedures"
        )
        logger.info("Indexing completed successfully")
        return chroma_db
    except Exception:
        logger.exception("Error creating embeddings or persisting to Chroma")
        return None