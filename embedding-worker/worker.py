import os
import time
import threading
import uvicorn
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, 
    CollectionStatus, UpdateStatus, CollectionInfo,
    VectorsConfig, OptimizersConfig, HnswConfig,
    OptimizersConfigDiff, HnswConfigDiff, UpdateCollection,
    CollectionParams
)
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydantic import BaseModel

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DOCS_PATH = os.environ.get("DOCS_PATH", "/documents")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
STATUS_PORT = int(os.environ.get("STATUS_PORT", "8001"))

# Initialize clients
client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(EMBEDDING_MODEL)

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

def chunk_text(text, chunk_size=500, overlap=100):
    # Simple word-based chunking
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def query_embeddings(query_text, collection_name="rag_collection", limit=5):
    """
    Query the Qdrant collection for documents similar to the query text.
    
    Args:
        query_text (str): The text to find similar documents for
        collection_name (str): Name of the Qdrant collection to search in
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of matching documents with their scores and metadata
    """
    try:
        print(f"\n🔍 Searching for: '{query_text}'")
        
        # Generate embedding for the query
        query_embedding = model.encode(query_text).tolist()
        
        # Search in Qdrant
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        # Process and format results
        results = []
        for i, hit in enumerate(search_results, 1):
            result = {
                'rank': i,
                'score': hit.score,
                'source': hit.payload.get('source', 'Unknown'),
                'text': hit.payload.get('text', '')[:200] + '...',  # Show first 200 chars
                'chunk_id': hit.payload.get('chunk_id', -1),
                'full_path': hit.payload.get('full_path', '')
            }
            results.append(result)
            
            # Print result summary
            print(f"\n📄 Result {i} (Score: {hit.score:.4f})")
            print(f"📂 Source: {result['source']}")
            print(f"🔗 Path: {result['full_path']}")
            print(f"📝 Text: {result['text']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error querying embeddings: {str(e)}")
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

def index_document(filepath, collection_name="rag_collection"):
    global is_indexing, current_file, processed_files, total_files, last_error
    
    # Initialize state
    last_error = None
    is_indexing = True
    current_file = os.path.basename(filepath)
    
    try:
        # Read and process the file
        print(f"Indexing {filepath}...")
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        # Generate chunks from the text
        chunks = chunk_text(text)
        
        # Skip empty files or files that couldn't be chunked
        if not chunks or not any(chunks):
            print(f"Warning: No valid content found in {filepath}")
            return
        
        # Filter out any empty chunks that might cause issues
        chunks = [chunk for chunk in chunks if chunk.strip()]
        if not chunks:
            print(f"Warning: No valid text chunks found in {filepath}")
            return
        
        # Generate embeddings for the chunks
        try:
            embeddings = model.encode(chunks)
            if len(embeddings) == 0 or (hasattr(embeddings, 'shape') and embeddings.shape[0] == 0):
                print(f"Warning: No embeddings generated for {filepath}")
                return
        except Exception as e:
            print(f"Error generating embeddings for {filepath}: {str(e)}")
            return
        
        # Get vector size from the first embedding
        vector_size = len(embeddings[0]) if len(embeddings) > 0 else 1024
        
        # Ensure collection exists with correct configuration
        collection_name = "rag_collection"
        ensure_collection_exists(collection_name, vector_size)
        
        # Create points for Qdrant
        points = []
        for i, (emb, chunk) in enumerate(zip(embeddings, chunks)):
            point = {
                "id": get_file_id(filepath, i),
                "vector": emb.tolist(),
                "payload": {
                    "source": os.path.basename(filepath),
                    "chunk_id": i,
                    "text": chunk,
                    "full_path": filepath,
                    "timestamp": int(time.time())
                }
            }
            points.append(point)
            print(f"Created point {i+1}/{len(chunks)} with vector size {len(emb)}")
        
        # First, delete existing chunks for this file if they exist
        delete_file_from_index(filepath, collection_name)
        
        # Upload points to Qdrant with retries
        max_retries = 3
        chunk_size = 100  # Increased from 10 to 100 for better performance
        
        print(f"Starting to process {len(points)} points in chunks of {chunk_size}")
        print(f"Vector size: {vector_size}")
        print(f"Collection name: {collection_name}")
        
        for attempt in range(max_retries):
            try:
                print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
                
                # Process points in chunks
                for i in range(0, len(points), chunk_size):
                    chunk_points = points[i:i + chunk_size]
                    chunk_num = i // chunk_size + 1
                    total_chunks = (len(points) - 1) // chunk_size + 1
                    
                    print(f"\nChunk {chunk_num}/{total_chunks} ({len(chunk_points)} points)")
                    
                    try:
                        # Print first point for debugging
                        first_point = chunk_points[0]
                        print(f"First point ID: {first_point['id']}")
                        print(f"First point vector length: {len(first_point['vector'])}")
                        
                        # Try to upsert the chunk
                        print(f"Upserting chunk {chunk_num}...")
                        operation_info = client.upsert(
                            collection_name=collection_name,
                            points=chunk_points,
                            wait=True
                        )
                        print(f"Successfully upserted chunk {chunk_num}")
                        
                        # Verify the points were added
                        count_result = client.count(
                            collection_name=collection_name,
                            count_filter=None
                        )
                        print(f"Current total points in collection: {count_result.count}")
                        
                    except Exception as chunk_error:
                        print(f"\n--- ERROR in chunk {chunk_num} ---")
                        print(f"Error type: {type(chunk_error).__name__}")
                        print(f"Error details: {str(chunk_error)}")
                        print("First point in failed chunk:")
                        print(f"ID: {chunk_points[0]['id']}")
                        raise  # Re-raise to trigger retry
                
                # If we get here, all chunks were processed successfully
                processed_files += 1
                total_files = max(total_files, processed_files)
                last_error = None
                print(f"✅ Successfully processed {len(points)} chunks from {filepath}")
                return  # Exit the function on success
                
            except Exception as e:
                last_error = str(e)
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries - 1:  # Last attempt
                    error_msg = f"Failed to index {filepath} after {max_retries} attempts: {str(e)}"
                    print(error_msg)
                    last_error = error_msg
                    break
                
                # Wait before retry with exponential backoff
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        # If we get here, all retries failed
        if last_error:
            raise Exception(last_error)
            
        return  # This should never be reached as we either return on success or raise an exception
        
    except Exception as e:
        # Handle any unexpected errors
        last_error = str(e)
        print(f"❌ Unexpected error processing {filepath}: {str(e)}")
        raise
        
    finally:
        # Always clean up
        is_indexing = False
        current_file = None

class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.txt', '.md')):
            index_document(event.src_path)

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
    
    # Get list of files in documents directory
    doc_path = Path(DOCS_PATH)
    all_files = [f for f in doc_path.glob('**/*') if f.is_file() and f.suffix.lower() in ('.txt', '.md')]
    
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
    
    # Watch for new files
    event_handler = DocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, DOCS_PATH, recursive=True)
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
