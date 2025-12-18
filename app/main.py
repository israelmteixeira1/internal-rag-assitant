import logging
from fastapi import FastAPI, HTTPException
from app.core.lifespan import lifespan
from app.schemas import HealthResponse, AnswerResponse, QuestionRequest
from app.services.rag_service import query_rag
from app.utils.text_utils import normalize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("main")

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