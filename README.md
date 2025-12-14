# RAG Assistant

## Overview
The RAG Assistant is a Python-based application designed to provide support for internal procedures through a Retrieval-Augmented Generation (RAG) approach. It connects to a static PDF document containing internal procedures and answers questions from the support team based on the content of that document.

## Project Structure
```
rag-assistant
├── .env
├── requirements.txt
├── rag_core.py
├── procedimentos.pdf
└── README.md
```

## Files Description
- **.env**: Contains environment variables, including a dummy entry for `GEMINI_API_KEY`.
- **requirements.txt**: Lists essential dependencies for the project:
  - langchain
  - langchain-google-genai
  - pypdf
  - chromadb
  - python-dotenv
  - fastapi
  - uvicorn
- **rag_core.py**: Implements the core functionality of the RAG assistant, including:
  - `funcao_indexar_pdf(caminho_pdf: str)`: Loads a PDF, splits the text, creates embeddings, and stores them in ChromaDB.
  - `funcao_consultar_rag(pergunta: str) -> str`: Connects to ChromaDB, retrieves relevant documents, and generates a response using ChatGoogleGenerativeAI.
- **procedimentos.pdf**: A dummy PDF containing at least 5 paragraphs simulating internal procedures, such as system restart, ticket opening, and checklists.
- **README.md**: Documentation for the project.

## Setup Instructions
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a `.env` file with your `GEMINI_API_KEY`.
4. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Ensure that the `procedimentos.pdf` file is present in the project directory.
2. Run the `rag_core.py` script to index the PDF and start querying:
   ```
   python rag_core.py
   ```
3. Use the function `funcao_consultar_rag(pergunta)` to ask questions based on the internal procedures.

## Notes
- Ensure that the PDF file is properly formatted and contains relevant information for accurate responses.
- Modify the `GEMINI_API_KEY` in the `.env` file to connect to the necessary APIs.