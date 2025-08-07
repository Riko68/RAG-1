import asyncio
import uuid
from fastapi import (
    FastAPI, Depends, Header, HTTPException, 
    UploadFile, File, status, Request, Query
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional, Dict, Any, Union
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import aiofiles
from redis.asyncio import Redis
from datetime import timedelta

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Import RAG service and memory
from .rag_service import RAGService
from .memory import MemoryManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --------- Configuration ---------
# Required environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL environment variable is required")

OLLAMA_URL = os.getenv("OLLAMA_URL")
if not OLLAMA_URL:
    raise ValueError("OLLAMA_URL environment variable is required")

# Optional environment variables with defaults
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MEMORY_TTL_HOURS = int(os.getenv("MEMORY_TTL_HOURS", "24"))  # Default 24 hours
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))  # Default 10 messages

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")
DOCS_PATH = os.getenv("DOCS_PATH", "/documents")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create documents directory if it doesn't exist
Path(DOCS_PATH).mkdir(parents=True, exist_ok=True)

# --------- Models ---------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="The question to ask the RAG system")
    top_k: Optional[int] = Field(3, ge=1, le=30, description="Number of relevant chunks to retrieve (max 30)")
    session_id: Optional[str] = Field(
        None, 
        description="Session ID for maintaining conversation context. If not provided, a new session will be created."
    )
    use_memory: Optional[bool] = Field(
        True, 
        description="Whether to use conversation memory for this request"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "top_k": 3,
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "use_memory": True
            }
        }
        
    @validator('session_id')
    def validate_session_id(cls, v):
        if v and not (len(v) == 36 and v.count('-') == 4):
            try:
                uuid.UUID(v)
                return v
            except ValueError:
                raise ValueError("session_id must be a valid UUID")
        return v

class RAGResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(..., description="List of source documents used for the answer")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for maintaining conversation context. A new ID is generated if not provided."
    )
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "The capital of France is Paris.",
                "sources": ["document1.txt", "document2.pdf"],
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
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
    version="1.0.0",
    # Ensure proper encoding for non-ASCII characters
    default_response_class=JSONResponse,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """
    Health check endpoint.
    
    Returns:
        dict: Status of all services
    """
    status = {
        "status": "ok",
        "services": {
            "qdrant": "ok" if rag_service and rag_service.qdrant_client else "error",
            "ollama": "ok" if rag_service and rag_service.llm else "error",
            "redis": "ok" if redis_client and rag_service.memory_manager else "disabled",
            "memory": "enabled" if rag_service and rag_service.memory_manager else "disabled"
        },
        "collection": COLLECTION_NAME,
        "model": rag_service.llm_model if rag_service else None
    }
    
    # Add more detailed status if available
    try:
        if rag_service and rag_service.qdrant_client:
            collections = rag_service.qdrant_client.get_collections()
            status["collections"] = [c.name for c in collections.collections]
    except Exception as e:
        status["services"]["qdrant"] = f"error: {str(e)}"
    
    return status

# Configure CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Add middleware to ensure proper request/response encoding
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers and ensure proper content type for responses."""
    response = await call_next(request)
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# Initialize Redis client
redis_client = None
try:
    redis_client = Redis.from_url(REDIS_URL)
    logger.info(f"Connected to Redis at {REDIS_URL}")
except Exception as e:
    logger.warning(f"Failed to connect to Redis: {str(e)}. Conversation memory will be disabled.")
    redis_client = None

# Initialize services
rag_service = None
memory_manager = MemoryManager(redis_client) if redis_client else None

# Initialize RAG chain on startup
@app.on_event("startup")
async def startup_event():
    global rag_service, memory_manager
    
    try:
        # Initialize RAG service with memory support
        rag_service = RAGService(
            qdrant_url=QDRANT_URL,
            ollama_url=OLLAMA_URL,
            redis_client=redis_client,
            collection_name=COLLECTION_NAME,
            memory_ttl_hours=MEMORY_TTL_HOURS,
            max_history_messages=MAX_HISTORY_MESSAGES
        )
        
        # Test Qdrant connection
        collections = rag_service.qdrant_client.get_collections()
        logger.info(f"Connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
        
        # Test Ollama connection
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: rag_service.llm("Test connection")
        )
        logger.info(f"Connected to Ollama at {OLLAMA_URL}")
        
        # Initialize rate limiter if Redis is available
        if redis_client:
            await FastAPILimiter.init(redis_client)
            logger.info("Rate limiter initialized with Redis")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
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
    request: QueryRequest,
    role: str = Depends(get_role),
    x_request_id: str = Header(None, description="Optional request ID for tracing")
):
    """
    Query the RAG system with a question, with optional conversation memory.
    
    This endpoint supports maintaining conversation context across multiple requests
    using a session ID. If no session ID is provided, a new conversation will be started.
    
    Example request with session:
    ```
    curl -X POST "http://localhost:8000/query" \
         -H "Content-Type: application/json" \
         -d '{
           "query": "What is the capital of France?",
           "session_id": "550e8400-e29b-41d4-a716-446655440000",
           "use_memory": true,
           "top_k": 3
         }'
    ```
    
    Example response:
    ```json
    {
      "answer": "The capital of France is Paris.",
      "sources": ["document1.pdf", "document2.txt"],
      "session_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    ```
    """
    logger_extra = {
        "request_id": x_request_id or str(uuid.uuid4()),
        "session_id": request.session_id or "new_session",
        "use_memory": request.use_memory,
        "top_k": request.top_k
    }
    
    try:
        logger.info(
            f"Query received (session: {request.session_id or 'new'})",
            extra=logger_extra
        )
        
        # Log the query request
        logger.info(f"Processing query: {query_request.query}")
        logger.info(f"Top K: {query_request.top_k}")
        
        # Get the RAG service response with timeout
        try:
            response = await asyncio.wait_for(
                ask_question(query_request, role),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            error_msg = "Query processing timed out after 30 seconds"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=error_msg
            )
        
        # Convert response to dict if it's a Pydantic model
        if hasattr(response, 'dict'):
            response = response.dict()
        
        # Log the results for debugging
        logger.info(f"Successfully processed query: {query_request.query}")
        if isinstance(response, dict):
            answer = response.get('answer', 'No answer found')
            logger.info(f"Answer length: {len(answer)} characters")
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
