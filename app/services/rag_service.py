import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.exceptions import LangChainException
from app.core import state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("query_rag")

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def query_rag(question: str) -> str:
    if state.db is None or state.llm is None:
        return "Erro interno: serviço indisponível."

    retriever = state.db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    docs = retriever.invoke(question)
    
    if not docs:
        return "Não consegui encontrar essa informação nos procedimentos internos."

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

    try:
        response = state.llm.invoke(
            prompt.format(
                context=format_docs(docs),
                question=question
            )
        )
        return response.content
    except LangChainException as e:
        logger.error("Erro no LLM: %s", e)
        return "O serviço de IA está temporariamente indisponível devido a limites de uso."
