from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
db: Optional[Chroma] = None
llm: Optional[ChatGoogleGenerativeAI] = None