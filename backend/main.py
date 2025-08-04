from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import aiofiles
import aioredis

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the capital of France?"
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure rate limiting
@app.on_event("startup")
async def startup():
    try:
        redis = Redis.from_url("redis://redis:6379", encoding="utf8", decode_responses=True)
        await FastAPILimiter.init(redis)
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize rate limiting"
        )

# --------- Role-based Access ---------
async def get_role(x_role: str = Header(...)):
    if x_role not in ["admin", "user"]:
        raise HTTPException(
            status_code=401,
            detail="Invalid role. Must be either 'admin' or 'user'"
        )
    return x_role

# --------- Initialize RAG chain ---------
async def initialize_rag_chain():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vectorstore = Qdrant(
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
        )
        llm = Ollama(base_url=OLLAMA_URL, model="mistral")
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff"
        )
        logger.info("RAG chain initialized successfully.")
        return rag_chain
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize RAG system"
        )

# Initialize RAG chain on startup
@app.on_event("startup")
async def startup_event():
    global rag_chain
    rag_chain = await initialize_rag_chain()

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
        if rag_chain is None:
            return JSONResponse(
                content={
                    "status": "warning",
                    "service": "RAG API",
                    "message": "RAG chain not initialized"
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

@app.post(
    "/ask",
    response_model=QueryRequest,
    summary="Ask a question using RAG",
    tags=["query"],
    dependencies=[Depends(RateLimiter(times=5, seconds=60))]
)
async def query_endpoint(query: QueryRequest, role: str = Depends(get_role)):
    """Ask a question using the RAG system. Requires a valid X-Role header."""
    try:
        if rag_chain is None:
            raise HTTPException(
                status_code=500,
                detail="RAG pipeline not initialized"
            )
            
        response = await asyncio.to_thread(rag_chain.run, query.query)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process query"
        )
