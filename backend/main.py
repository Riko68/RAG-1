from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
import os
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import aiofiles
from redis.asyncio import Redis

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Import RAG service
from rag_service import RAGService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --------- Configuration ---------
QDRANT_URL = os.getenv("QDRANT_URL")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL environment variable is required")

OLLAMA_URL = os.getenv("OLLAMA_URL")
if not OLLAMA_URL:
    raise ValueError("OLLAMA_URL environment variable is required")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
DOCS_PATH = os.getenv("DOCS_PATH", "/documents")

# Create documents directory if it doesn't exist
Path(DOCS_PATH).mkdir(parents=True, exist_ok=True)

# --------- Models ---------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="The question to ask the RAG system")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of relevant chunks to retrieve")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "top_k": 3
            }
        }

class RAGResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(..., description="List of source documents used for the answer")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "The capital of France is Paris.",
                "sources": ["document1.txt", "document2.pdf"]
            }
        }

class DocumentUploadResponse(BaseModel):
    filename: str
    status: str
    message: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "filename": "example.txt",
                "status": "success",
                "message": "Document uploaded successfully"
            }
        }

class DocumentListResponse(BaseModel):
    documents: List[str]

    class Config:
        schema_extra = {
            "example": {
                "documents": ["example.txt", "example2.md"]
            }
        }

# --------- App ---------
app = FastAPI(
    title="RAG API",
    description="API for Retrieval Augmented Generation system",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_service = None

# Initialize RAG chain on startup
@app.on_event("startup")
async def startup_event():
    global rag_service
    
    # Initialize rate limiter
    redis = Redis.from_url("redis://redis:6379/0")
    await FastAPILimiter.init(redis)
    logger.info("Rate limiter initialized")
    
    # Initialize RAG service
    try:
        # Initialize RAG service with the same model as the embedding worker
        rag_service = RAGService(
            qdrant_url=QDRANT_URL,
            ollama_url=OLLAMA_URL,
            collection_name=COLLECTION_NAME,
            model_name="BAAI/bge-m3"  # Match the embedding worker's model
        )
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {str(e)}")
        raise

# --------- Role-based Access ---------
async def get_role(x_role: str = Header(...)):
    if x_role not in ["admin", "user"]:
        raise HTTPException(
            status_code=401,
            detail="Invalid role. Must be either 'admin' or 'user'"
        )
    return x_role

# --------- Endpoints ---------
@app.get(
    "/health",
    summary="Check API health",
    tags=["health"]
)
async def health():
    """Check if the API is running and healthy."""
    try:
        # Check if RAG chain is initialized
        if rag_service is None:
            return JSONResponse(
                content={
                    "status": "warning",
                    "service": "RAG API",
                    "message": "RAG service not initialized"
                },
                status_code=200
            )
            
        # Check Qdrant connection
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
                if response.status_code != 200:
                    return JSONResponse(
                        content={
                            "status": "warning",
                            "service": "RAG API",
                            "message": "Qdrant connection issue"
                        },
                        status_code=200
                    )
        except Exception as e:
            return JSONResponse(
                content={
                    "status": "warning",
                    "service": "RAG API",
                    "message": f"Failed to connect to Qdrant: {str(e)}"
                },
                status_code=200
            )
            
        return JSONResponse(
            content={
                "status": "ok",
                "service": "RAG API",
                "message": "All services healthy"
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "service": "RAG API",
                "message": "Health check failed"
            },
            status_code=500
        )

@app.post(
    "/admin/reindex",
    summary="Trigger full reindex of documents",
    tags=["admin"],
    dependencies=[Depends(RateLimiter(times=1, seconds=3600))]
)
async def reindex(role: str = Depends(get_role)):
    """Trigger a full reindex of all documents. Only available to admin users."""
    if role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required for reindexing"
        )
    try:
        # In real implementation: Trigger reindexing logic or worker signal here
        return JSONResponse(
            content={"message": "Reindex triggered. This may take some time."},
            status_code=202
        )
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to trigger reindex"
        )

@app.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all available documents",
    tags=["documents"],
    dependencies=[Depends(RateLimiter(times=10, seconds=60))]
)
async def list_documents():
    """List all documents available in the documents directory."""
    try:
        if not os.path.exists(DOCS_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"Documents folder not found at {DOCS_PATH}"
            )
            
        files = os.listdir(DOCS_PATH)
        return DocumentListResponse(documents=files)
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list documents"
        )

@app.post("/ask", response_model=RAGResponse, status_code=status.HTTP_200_OK)
async def ask_question(
    query: QueryRequest,
    role: str = Depends(get_role)
):
    """
    Ask a question using the RAG system.
    
    This endpoint retrieves relevant context from the vector database and uses
    Ollama to generate a response based on that context.
    
    - **query**: The question to ask
    - **top_k**: Number of relevant chunks to retrieve (default: 3)
    """
    if not rag_service:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG service is not initialized"
        )
    
    try:
        logger.info(f"Processing question: {query.query}")
        
        # Get response from RAG service
        response = await rag_service.ask(
            query=query.query,
            top_k=query.top_k
        )
        
        logger.info(f"Successfully generated response for question: {query.query}")
        return response
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your question: {str(e)}"
        )

# Keep the old query endpoint for backward compatibility
@app.post("/query", response_model=RAGResponse)
async def query_endpoint(
    request: dict = {
        "query": "What is the capital of France?",
        "top_k": 3
    },
    role: str = Header("user", description="User role (user, admin)")
):
    """
    Query the RAG system with a question.
    
    This is a simplified endpoint that doesn't require the full QueryRequest model.
    It will automatically add debug logging of the search results.
    
    Example request:
    ```
    curl -X POST "http://localhost:8000/query" \
         -H "Content-Type: application/json" \
         -d '{"query": "Your question here"}'
    """
    try:
        # Convert to QueryRequest
        query_request = QueryRequest(
            query=request.get("query", ""),
            top_k=request.get("top_k", 3)
        )
        
        # Log the query
        logger.info(f"Processing query: {query_request.query}")
        
        # Get the RAG service response
        response = await ask_question(query_request, role)
        
        # Convert response to dict if it's a Pydantic model
        if hasattr(response, 'dict'):
            response = response.dict()
        
        # Log the results for debugging
        logger.info(f"Query: {query_request.query}")
        logger.info(f"Response: {response}")
        
        if isinstance(response, dict):
            logger.info(f"Answer: {response.get('answer', 'No answer found')}")
            sources = response.get('sources', [])
            if sources:
                logger.info(f"Sources used: {', '.join(sources) if isinstance(sources, list) else sources}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your query: {str(e)}"
        )
