import os
import time
import threading
import uvicorn
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, 
    CollectionStatus, UpdateStatus, CollectionInfo,
    VectorsConfig, OptimizersConfig, HnswConfig,
    OptimizersConfigDiff, HnswConfigDiff, UpdateCollection,
    CollectionParams, Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydantic import BaseModel

# Import the document processor
from document_processor import DocumentProcessor, DocumentChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DOCS_PATH = os.environ.get("DOCS_PATH", "/documents")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/uploads")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/processed")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
STATUS_PORT = int(os.environ.get("STATUS_PORT", "8001"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize clients and processors
client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(EMBEDDING_MODEL)
document_processor = DocumentProcessor(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# FastAPI app for status endpoint
app = FastAPI(
    title="RAG Embedding Worker API",
    description="API for document embedding and search",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
is_indexing = False
current_file = None
processed_files = 0
total_files = 0
last_error = None

# Response Models
class SearchResult(BaseModel):
    rank: int
    score: float
    source: str
    text: str
    chunk_id: int
    full_path: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
        
class StatusResponse(BaseModel):
    status: str
    current_file: Optional[str]
    processed_files: int
    total_files: int
    last_error: Optional[str]
    timestamp: int
    qdrant_status: str
    model_name: str

@app.get("/status", response_model=StatusResponse, summary="Get current indexing status")
async def get_status():
    """Get the current status of the embedding worker"""
    global is_indexing, current_file, processed_files, total_files, last_error
    
    try:
        # Check Qdrant connection
        client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
    
    return {
        "status": "indexing" if is_indexing else "idle",
        "current_file": current_file,
        "processed_files": processed_files,
        "total_files": total_files,
        "last_error": last_error,
        "timestamp": int(time.time()),
        "qdrant_status": qdrant_status,
        "model_name": EMBEDDING_MODEL
    }

@app.get("/search", response_model=SearchResponse, summary="Search documents")
async def search_documents(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, description="Maximum number of results to return", ge=1, le=20)
):
    """Search for documents matching the query"""
    try:
        results = query_embeddings(q, limit=limit)
        return {
            "query": q,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", status_code=202, summary="Index a document from URL")
async def index_document_url(url: str):
    """Trigger indexing of a document from a URL"""
    try:
        # In a real implementation, you would download the file from the URL
        # For now, we'll just print a message
        print(f"Received request to index document from URL: {url}")
        return {"status": "accepted", "url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a simple root endpoint with API documentation link
@app.get("/")
async def root():
    return {
        "message": "RAG Embedding Worker API",
        "docs": "/docs",
        "status": "/status",
        "search": "/search?q=your+query"
    }

def run_status_server():
    """Run the status server in a separate thread"""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=STATUS_PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()

def process_document_chunks(chunks: List[DocumentChunk], filepath: str) -> Tuple[List[PointStruct], int]:
    """Process document chunks into Qdrant points.
    
    Args:
        chunks: List of DocumentChunk objects
        filepath: Path to the source file
        
    Returns:
        Tuple of (points, number of chunks)
    """
    points = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Generate embedding for the chunk
            embedding = model.encode(chunk.text, normalize_embeddings=True).tolist()
            
            # Create a point for Qdrant
            point_id = get_file_id(filepath, i)
            
            # Prepare metadata
            metadata = chunk.metadata.copy()
            metadata.update({
                'chunk_id': i,
                'file_modified': int(os.path.getmtime(filepath)),
                'processed_at': int(time.time())
            })
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata
            )
            points.append(point)
            
        except Exception as e:
            logger.error(f"Error processing chunk {i} of {filepath}: {str(e)}")
            continue
    
    return points, len(chunks)

def query_embeddings(query_text: str, collection_name: str = "rag_collection", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Query the Qdrant collection for documents similar to the query text.
    
    Args:
        query_text: The text to find similar documents for
        collection_name: Name of the Qdrant collection to search in
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents with their scores and metadata
    """
    try:
        # Generate embedding for the query
        query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()
        
        # Search the collection
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_vectors=False,
            with_payload=True,
            score_threshold=0.3  # Lower threshold for better recall
        )
        
        # Format results
        results = []
        for i, hit in enumerate(search_results):
            payload = hit.payload or {}
            results.append({
                'rank': i + 1,
                'score': hit.score,
                'source': payload.get('source', 'unknown'),
                'text': payload.get('text', ''),
                'chunk_id': payload.get('chunk_id', -1),
                'full_path': payload.get('full_path', ''),
                'chunk_type': payload.get('chunk_type', 'text')
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error querying embeddings: {str(e)}")
        raise

def get_file_id(filepath, chunk_idx):
    """Generate a consistent integer ID from file path and chunk index"""
    import hashlib
    # Create a unique string from filepath and chunk index
    unique_str = f"{os.path.abspath(filepath)}_{chunk_idx}"
    # Generate a hash and convert to integer (first 8 bytes)
    return int(hashlib.md5(unique_str.encode()).hexdigest()[:8], 16) % (10**8)

def ensure_collection_exists(collection_name, vector_size=1024):
    """Ensure the Qdrant collection exists with proper configuration"""
    from qdrant_client.models import Distance, VectorParams
    
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            print(f"Creating new collection: {collection_name}")
            
            # Create collection with optimized settings
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
                optimizers_config={
                    'indexing_threshold': 0,  # Index all vectors
                    'memmap_threshold': 20000,
                    'default_segment_number': 1,
                    'max_optimization_threads': 4
                },
                hnsw_config={
                    'm': 16,
                    'ef_construct': 100,
                    'full_scan_threshold': 10,
                    'on_disk': False
                }
            )
            print(f"Successfully created collection: {collection_name}")
        else:
            print(f"Using existing collection: {collection_name}")
            
    except Exception as e:
        print(f"Error managing collection: {str(e)}")
        raise

def get_indexed_files(collection_name="rag_collection"):
    """Get a dictionary of indexed files and their latest modification timestamps"""
    try:
        # Get all points with their full_path and timestamp payloads
        points, _ = client.scroll(
            collection_name=collection_name,
            with_payload=["full_path", "timestamp"],
            limit=10000,  # Adjust based on your expected number of files
            with_vectors=False
        )
        
        # Track the latest timestamp for each file
        file_timestamps = {}
        for point in points:
            filepath = point.payload.get("full_path")
            timestamp = point.payload.get("timestamp", 0)
            
            # Keep the latest timestamp for each file
            if filepath and (filepath not in file_timestamps or timestamp > file_timestamps[filepath]):
                file_timestamps[filepath] = timestamp
                
        return file_timestamps
        
    except Exception as e:
        print(f"Error getting indexed files: {str(e)}")
        return {}

def needs_reindex(filepath, indexed_files):
    """Check if a file needs to be reindexed"""
    if not os.path.exists(filepath):
        return False  # File no longer exists
        
    if filepath not in indexed_files:
        return True  # New file, needs indexing
        
    try:
        # Check if file has been modified since last index
        file_mtime = int(os.path.getmtime(filepath))
        last_indexed = indexed_files[filepath]
        return file_mtime > last_indexed
    except (OSError, KeyError) as e:
        print(f"Error checking file {filepath}: {str(e)}")
        return True

def delete_file_from_index(filepath, collection_name="rag_collection"):
    """Delete all chunks for a file from the index"""
    try:
        # Delete all points with the matching full_path
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="full_path",
                            match=models.MatchValue(value=filepath)
                        )
                    ]
                )
            )
        )
        print(f"Deleted all chunks for {filepath} from index")
        return True
    except Exception as e:
        print(f"Error deleting {filepath} from index: {str(e)}")
        return False

def index_document(filepath: str, collection_name: str = "rag_collection") -> bool:
    """Index a single document file with enhanced processing.
    
    Args:
        filepath: Path to the document to index
        collection_name: Name of the Qdrant collection
        
    Returns:
        bool: True if indexing was successful, False otherwise
    """
    global is_indexing, current_file, processed_files, total_files, last_error
    
    if not os.path.exists(filepath):
        last_error = f"File not found: {filepath}"
        logger.error(last_error)
        return
    
    try:
        is_indexing = True
        current_file = filepath
        logger.info(f"Processing document: {filepath}")
        
        # Process the document using DocumentProcessor
        chunks = document_processor.process_file(filepath)
        
        # Process chunks into Qdrant points
        points, num_chunks = process_document_chunks(chunks, filepath)
        
        if not points:
            logger.warning(f"No valid chunks generated from {filepath}")
            return
        
        # Delete existing points for this file
        delete_file_from_index(filepath, collection_name)
        
        # Ensure collection exists with the correct vector size
        vector_size = len(points[0].vector) if points else 1024
        ensure_collection_exists(collection_name, vector_size)
        
        # Process points in chunks to avoid timeouts
        max_retries = 3
        chunk_size = 50  # Process points in smaller chunks for reliability
        
        logger.info(f"Processing {len(points)} points in chunks of {chunk_size}")
        
        for i in range(0, len(points), chunk_size):
            chunk_points = points[i:i + chunk_size]
            
            for attempt in range(max_retries):
                try:
                    # Upsert the chunk of points
                    client.upsert(
                        collection_name=collection_name,
                        points=chunk_points,
                        wait=True
                    )
                    logger.info(f"Successfully indexed chunk {i//chunk_size + 1}/{(len(points)-1)//chunk_size + 1}")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying: {str(e)}")
                    time.sleep(1)  # Wait before retry
        
        processed_files += 1
        logger.info(f"Successfully indexed {num_chunks} chunks from {filepath}")
        
    except Exception as e:
        last_error = f"Error processing {filepath}: {str(e)}"
        logger.error(last_error, exc_info=True)
        raise
    finally:
        current_file = None
        is_indexing = False
                
        # Process points in chunks to avoid timeouts
        max_retries = 3
        chunk_size = 50  # Process points in smaller chunks for reliability
        
        logger.info(f"Processing {len(points)} points in chunks of {chunk_size}")
        
        for i in range(0, len(points), chunk_size):
            chunk_points = points[i:i + chunk_size]
            
            for attempt in range(max_retries):
                try:
                    # Upsert the chunk of points
                    client.upsert(
                        collection_name=collection_name,
                        points=chunk_points,
                        wait=True
                    )
                    logger.info(f"Successfully indexed chunk {i//chunk_size + 1}/{(len(points)-1)//chunk_size + 1}")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying: {str(e)}")
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        processed_files += 1
        total_files = max(total_files, processed_files)
        logger.info(f"Successfully indexed {num_chunks} chunks from {filepath}")

class DocumentHandler(FileSystemEventHandler):
    """Handles file system events for document processing."""
    
    # Supported file extensions and their MIME types
    SUPPORTED_EXTENSIONS = {
        # Text files
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        
        # Document files
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        
        # Add more supported formats as needed
    }
    
    def on_created(self, event):
        """Handle file creation events in both upload and documents directories."""
        if event.is_directory:
            return
            
        filepath = event.src_path
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext in self.SUPPORTED_EXTENSIONS:
            try:
                # Check if the file is ready (not being written to)
                if not self._is_file_ready(filepath):
                    logger.warning(f"File {filepath} is not ready for processing, will retry")
                    return
                    
                logger.info(f"Detected new file: {filepath}")
                
                # If file is in uploads, process and move it
                if filepath.startswith(UPLOAD_DIR):
                    if index_document(filepath):
                        self._move_to_processed(filepath)
                else:
                    # Process files in documents directory without moving
                    index_document(filepath)
                
            except Exception as e:
                logger.error(f"Error processing {filepath}: {str(e)}", exc_info=True)
    
    def _move_to_processed(self, filepath):
        """Move a file from uploads to processed directory."""
        try:
            # Create relative path to preserve directory structure
            rel_path = os.path.relpath(filepath, UPLOAD_DIR)
            dest_path = os.path.join(PROCESSED_DIR, rel_path)
            
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Move the file
            os.rename(filepath, dest_path)
            logger.info(f"Moved processed file to: {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving file {filepath} to processed: {str(e)}")
            return False
    
    def _is_file_ready(self, filepath, max_attempts=5, delay=1):
        """Check if a file is ready to be processed."""
        for attempt in range(max_attempts):
            try:
                # Try to open the file exclusively
                with open(filepath, 'rb') as f:
                    # Read a small chunk to check if the file is accessible
                    f.read(1024)
                return True
            except (IOError, PermissionError) as e:
                if attempt == max_attempts - 1:
                    return False
                time.sleep(delay)
        return False

def interactive_query_mode():
    """Run an interactive query mode for testing document search"""
    print("\n" + "="*50)
    print("  DOCUMENT SEARCH MODE")
    print("  Type 'exit' to quit")
    print("  Type 'status' to see indexing status")
    print("="*50 + "\n")
    
    while True:
        try:
            query = input("\n🔎 Enter your search query: ").strip()
            
            if query.lower() == 'exit':
                print("Exiting search mode...")
                break
                
            if query.lower() == 'status':
                print(f"\n📊 Indexing Status:")
                print(f"- Current file: {current_file or 'None'}")
                print(f"- Processed files: {processed_files}")
                print(f"- Total files: {total_files}")
                if last_error:
                    print(f"- Last error: {last_error}")
                continue
                
            if not query:
                continue
                
            # Execute the search
            query_embeddings(query)
            
        except KeyboardInterrupt:
            print("\nExiting search mode...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Document Embedding Worker')
    parser.add_argument('--query', action='store_true', 
                       help='Run in interactive query mode')
    parser.add_argument('--search', type=str, 
                       help='Run a single search query and exit')
    args = parser.parse_args()
    
    # Start the status server in a separate thread
    status_thread = threading.Thread(target=run_status_server, daemon=True)
    status_thread.start()
    
    # Handle query mode if requested
    if args.search:
        query_embeddings(args.search)
        sys.exit(0)
    elif args.query:
        interactive_query_mode()
        sys.exit(0)
    
    # Default behavior: start file watcher and index existing files
    print(f"Starting document indexer in {DOCS_PATH}...")
    
    # Get list of currently indexed files with their timestamps
    print("Checking for indexed files in Qdrant...")
    indexed_files = get_indexed_files()
    print(f"Found {len(indexed_files)} files in the index")
    
    # Get list of files in both documents and uploads directories
    doc_path = Path(DOCS_PATH)
    upload_path = Path(UPLOAD_DIR)
    
    # Get files from documents directory
    all_files = [f for f in doc_path.glob('**/*') if f.is_file() and f.suffix.lower() in DocumentHandler.SUPPORTED_EXTENSIONS]
    
    # Get files from uploads directory
    if upload_path.exists():
        all_files.extend([f for f in upload_path.glob('**/*') if f.is_file() and f.suffix.lower() in DocumentHandler.SUPPORTED_EXTENSIONS])
    
    # Find files that need indexing (new or modified)
    files_to_index = [
        str(f) for f in all_files 
        if needs_reindex(str(f), indexed_files)
    ]
    
    # Find files that were deleted but still in the index
    indexed_file_paths = set(indexed_files.keys())
    current_file_paths = {str(f) for f in all_files}
    deleted_files = indexed_file_paths - current_file_paths
    
    # Clean up deleted files from the index
    for deleted_file in deleted_files:
        print(f"File deleted, removing from index: {deleted_file}")
        delete_file_from_index(deleted_file)
    
    total_files = len(files_to_index)
    processed_files = 0
    
    if total_files > 0:
        print(f"Found {total_files} files to index (new or modified)")
        
        # Index files that need it
        for fpath in files_to_index:
            index_document(fpath)
    else:
        print("No files need indexing - all files are up to date")
    
    # Clean up any empty collections that might exist
    try:
        collections = client.get_collections()
        for collection in collections.collections:
            if collection.name != "rag_collection":
                try:
                    client.delete_collection(collection.name)
                    print(f"Cleaned up unused collection: {collection.name}")
                except Exception as e:
                    print(f"Error cleaning up collection {collection.name}: {str(e)}")
    except Exception as e:
        print(f"Error checking collections: {str(e)}")
    
    # Watch for new files in both directories
    event_handler = DocumentHandler()
    observer = Observer()
    
    # Watch documents directory
    observer.schedule(event_handler, DOCS_PATH, recursive=True)
    
    # Watch uploads directory if it exists
    if os.path.exists(UPLOAD_DIR):
        observer.schedule(event_handler, UPLOAD_DIR, recursive=True)
    else:
        logger.warning(f"Upload directory not found: {UPLOAD_DIR}")
        
    observer.start()
    
    print(f"Watching {DOCS_PATH} for new documents...")
    print(f"Status available at http://localhost:{STATUS_PORT}/status")
    print("\nUse --query to start interactive search mode")
    print("or --search 'your query' to search directly")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        observer.stop()
    observer.join()
