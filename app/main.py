import logging
from fastapi import FastAPI, HTTPException
from app.core.lifespan import lifespan
from app.schemas import AnswerResponse, QuestionRequest
from app.services.rag_service import query_rag
from app.utils.text_utils import normalize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("main")

app = FastAPI(
    title="RAG Assistant API",
    description="API para consultar procedimentos internos usando RAG",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint principal para perguntas.
    Recebe uma pergunta e retorna a resposta baseada nos procedimentos internos.
    """
    if not request.question or not request.question.strip():
        logger.warning("Pergunta vazia ou inválida recebida")
        raise HTTPException(
            status_code=400,
            detail={"error": "Pergunta vazia ou inválida"}
        )
    
    question = normalize_text(request.question.strip())
    answer = query_rag(question)
    
    if answer is None:
        logger.error("Falha ao gerar a resposta")
        raise HTTPException(
            status_code=500,
            detail={"error": "Erro interno ao gerar a resposta"}
        )
    
    return AnswerResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)