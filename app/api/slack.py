import logging
import httpx
from fastapi import APIRouter, Request, Form
from fastapi.responses import JSONResponse
from app.services.rag_service import query_rag

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/slack", tags=["Slack"])

@router.post("/command")
async def slack_command(
    request: Request,
    text: str = Form(...),
    response_url: str = Form(...)
):
    """
    Endpoint para Slash Command do Slack.
    Exibe a pergunta enviada e depois envia a resposta final via response_url.
    """

    logger.info("Requisição recebida do Slack")
    logger.info(f"Pergunta recebida: {text}")

    placeholder_response = {
        "response_type": "ephemeral",
        "text": f"*Pergunta:* {text}\n\n🤖 Processando sua resposta..."
    }

    async def send_final_response():
        resposta_final = query_rag(text)
        logger.info("Resposta final da RAG pronta, enviando ao Slack")
        async with httpx.AsyncClient() as client:
            await client.post(response_url, json={
                "response_type": "ephemeral",
                "text": f"*Pergunta:* {text}\n*Resposta:* {resposta_final}"
            })
        logger.info("Resposta final enviada com sucesso")

    import asyncio
    asyncio.create_task(send_final_response())

    logger.info("Resposta placeholder enviada")
    return JSONResponse(content=placeholder_response)
