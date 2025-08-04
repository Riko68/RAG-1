from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- Configuration ---------
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
COLLECTION_NAME = "your_collection"  # Adjust to match your embedding worker setup
DOCS_PATH = os.getenv("DOCS_PATH", "/documents")

# --------- Models ---------
class QueryRequest(BaseModel):
    query: str

# --------- App ---------
app = FastAPI()

# --------- Role-based Access ---------
def get_role(x_role: str = Header(...)):
    if x_role not in ["admin", "user"]:
        raise HTTPException(status_code=401, detail="Invalid role")
    return x_role

# --------- Initialize RAG chain ---------
try:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Qdrant(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )
    llm = Ollama(base_url=OLLAMA_URL, model="mistral")
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    logger.info("RAG chain initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize RAG chain: {e}")
    rag_chain = None

# --------- Endpoints ---------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/admin/reindex")
def reindex(role: str = Depends(get_role)):
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    # In real implementation: Trigger reindexing logic or worker signal here
    return {"message": "Reindex triggered."}

@app.get("/documents")
def list_documents():
    if not os.path.exists(DOCS_PATH):
        raise HTTPException(status_code=500, detail=f"Documents folder not found at {DOCS_PATH}")
    try:
        files = os.listdir(DOCS_PATH)
        return {"documents": files}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents.")

@app.post("/ask")
def query_endpoint(query: QueryRequest):
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized.")
    try:
        response = rag_chain.run(query.query)
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query.")
