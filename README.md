# RAG Assistant

## Overview

The RAG Assistant is a Python-based application designed to provide support for internal procedures through a Retrieval-Augmented Generation (RAG) approach. It connects to a static PDF document containing internal procedures and answers questions from the support team based on the content of that document.

The application uses:
- **LangChain** for document processing and RAG pipeline
- **Google Gemini** for embeddings and LLM responses
- **ChromaDB** as the vector database
- **FastAPI** for HTTP API endpoints

## Project Structure

```
rag-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── schemas.py              # Pydantic models for request/response
│   ├── core/
│   │   ├── __init__.py
│   │   ├── lifespan.py         # Application lifecycle management
│   │   └── state.py            # Global state (db, llm, embeddings)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── indexing_service.py # PDF indexing logic
│   │   └── rag_service.py      # RAG query logic
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Centralized logging configuration
│       └── text_utils.py       # Text normalization utilities
├── chroma_db/                  # Vector database (auto-generated)
├── procedures.pdf              # PDF with internal procedures
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables
└── README.md                   # Project documentation
```

## Core Functions

### `indexing_service.py`

- **`index_pdf(pdf_path: str, embeddings_instance)`**: Loads a PDF file, splits the text into chunks, creates embeddings using Google's text-embedding-004 model, and stores them in ChromaDB.

### `rag_service.py`

- **`query_rag(question: str) -> str`**: Retrieves relevant document chunks from ChromaDB based on the question and generates a response using ChatGoogleGenerativeAI (Gemini 2.0 Flash).

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core LangChain framework |
| `langchain-community` | Document loaders (PyPDFLoader) |
| `langchain-google-genai` | Google Gemini integration |
| `langchain-chroma` | ChromaDB vector store integration |
| `langchain-text-splitters` | Text chunking utilities |
| `chromadb` | Vector database |
| `pypdf` | PDF parsing |
| `python-dotenv` | Environment variable management |
| `fastapi` | HTTP API framework |
| `uvicorn` | ASGI server |

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd rag-assistant
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> **Note:** Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 5. Add your PDF document

Place your internal procedures PDF file named `procedures.pdf` in the project root directory.

## Usage

### Running the API Server

```bash
# Using Python directly
python -m app.main

# Or using Uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I restart the system?"}'
```

**Response:**
```json
{"answer": "To restart the system, follow these steps: 1. Save all open files..."}
```

### Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Empty or invalid question |
| 500 | Internal error generating response |

## API Documentation

Once the server is running, access the interactive API documentation at:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Notes

- Ensure that the PDF file is properly formatted and contains relevant information for accurate responses.
- The ChromaDB vector database is automatically created on first run and persisted in the `chroma_db/` directory.
- Subsequent runs will load the existing database unless you delete the `chroma_db/` folder.
- Update the `GOOGLE_API_KEY` in the `.env` file with a valid Google AI API key.
- The application uses Gemini 2.0 Flash for fast, cost-effective responses.

## License

MIT License