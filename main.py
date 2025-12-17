from dotenv import load_dotenv
from contextlib import asynccontextmanager

import os
import re
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.exceptions import LangChainException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("rag_api")


# Variáveis Globais (inicializadas no lifespan)
embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
db: Optional[Chroma] = None
llm: Optional[ChatGoogleGenerativeAI] = None


class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

class HealthResponse(BaseModel):
    status: str


def normalize_text(text: str) -> str:
    """Normaliza texto removendo quebras de linha excessivas e espaços."""
    try:
        text = re.sub(r'\n{2,}', '\n\n', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'(\d+)\s+\.', r'\1.', text)
        text = text.replace(' .', '.').replace(' ,', ',')
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    except Exception:
        logger.exception("Error normalizing text")
        return text

def format_docs(docs: List[Document]) -> str:
    """Formata lista de documentos em uma única string."""
    try:
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception:
        logger.exception("Error formatting documents")
        return ""

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


def query_rag(question: str) -> str:
    """Executa a consulta RAG e retorna a resposta."""
    logger.info("Received question: %s", question)

    if db is None or llm is None:
        logger.error("RAG services (DB/LLM) not initialized.")
        return "Internal error: service unavailable."

    try:
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )
        docs = retriever.invoke(question)
        logger.info("Retrieved %d relevant chunks", len(docs))
    except Exception:
        logger.exception("Error retrieving documents")
        return "Error retrieving internal documents."

    if not docs:
        return "I couldn't find this information in the internal procedures."

    try:
        context = format_docs(docs)

        prompt = ChatPromptTemplate.from_template("""
You are an internal support assistant.

You MUST answer strictly and exclusively using the information present in the context below.
Do NOT use prior knowledge.
Do NOT make assumptions.
Do NOT invent information.

If the answer is not explicitly present in the context, reply exactly with:
"I couldn't find this information in the internal procedures."

Context:
{context}

Question:
{question}
""")

        response = llm.invoke(
            prompt.format(
                context=context,
                question=question
            )
        )
        logger.info("Response generated successfully")
        return response.content
    except LangChainException as e:
      logger.error("LLM quota or generation error: %s", e)
      return "The AI service is temporarily unavailable due to usage limits. Please try again later."

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
    global embeddings, db, llm
    
    # 1. Carregar variáveis de ambiente
    load_dotenv()
    logger.info("Environment variables loaded")
    
    # 2. Inicializar Embeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        logger.info("Embeddings initialized successfully")
    except Exception:
        logger.critical("Failed to initialize Embeddings. Check your credentials.")
        raise
    
    # 3. Inicializar LLM
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, request_timeout=20)
        logger.info("LLM initialized successfully")
    except Exception:
        logger.critical("Failed to initialize LLM. Check your credentials.")
        raise
    
    # 4. Carregar ou criar Chroma DB
    chroma_db_dir = "./chroma_db"
    if os.path.isdir(chroma_db_dir):
        try:
            db = Chroma(
                persist_directory=chroma_db_dir,
                collection_name="internal_procedures",
                embedding_function=embeddings
            )
            logger.info("Chroma DB loaded from disk successfully")
        except Exception:
            logger.warning("Error loading Chroma DB. Attempting to reindex...")
            db = index_pdf("procedures.pdf", embeddings)
    else:
        logger.info("Chroma DB not found. Indexing PDF...")
        db = index_pdf("procedures.pdf", embeddings)
    
    if db is None:
        logger.critical("Critical failure during indexing. Application cannot start.")
        raise RuntimeError("Failed to initialize Chroma DB")
    
    logger.info("RAG API ready to receive requests")
    
    yield
    
    logger.info("Shutting down RAG API")


app = FastAPI(
    title="RAG Assistant API",
    description="API for querying internal procedures using RAG",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Retorna status da API.
    """
    return HealthResponse(status="ok")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint principal para perguntas.
    Recebe uma pergunta e retorna a resposta baseada nos procedimentos internos.
    """
    if not request.question or not request.question.strip():
        logger.warning("Empty or invalid question received")
        raise HTTPException(
            status_code=400,
            detail={"error": "Empty or invalid question"}
        )
    
    question = normalize_text(request.question.strip())
    answer = query_rag(question)
    
    # Tratamento de erro interno
    if answer is None:
        logger.error("Failed to generate response")
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal error while generating response"}
        )
    
    return AnswerResponse(answer=answer)

# Execução direta (desenvolvimento)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)