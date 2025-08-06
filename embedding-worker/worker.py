import os
import time
import threading
import uvicorn
from typing import Optional
from fastapi import FastAPI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydantic import BaseModel

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DOCS_PATH = os.environ.get("DOCS_PATH", "/documents")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
STATUS_PORT = int(os.environ.get("STATUS_PORT", "8001"))

# Global state for status tracking
is_indexing = False
current_file = None
total_files = 0
processed_files = 0
last_error = None

# Initialize clients
client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(EMBEDDING_MODEL)

# FastAPI app for status endpoint
app = FastAPI(title="Embedding Worker Status")

class StatusResponse(BaseModel):
    is_indexing: bool
    current_file: Optional[str]
    processed_files: int
    total_files: int
    last_error: Optional[str]
    qdrant_status: str
    model_name: str

@app.get("/status")
async def get_status():
    """Get current status of the embedding worker"""
    global is_indexing, current_file, processed_files, total_files, last_error
    
    try:
        # Verify Qdrant connection
        collections = client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
    
    return StatusResponse(
        is_indexing=is_indexing,
        current_file=current_file,
        processed_files=processed_files,
        total_files=total_files,
        last_error=last_error,
        qdrant_status=qdrant_status,
        model_name=EMBEDDING_MODEL
    )

def run_status_server():
    """Run the status server in a separate thread"""
    uvicorn.run(app, host="0.0.0.0", port=STATUS_PORT)

def chunk_text(text, chunk_size=500, overlap=100):
    # Simple word-based chunking
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def get_file_id(filepath, chunk_idx):
    """Generate a consistent integer ID from file path and chunk index"""
    import hashlib
    # Create a unique string from filepath and chunk index
    unique_str = f"{os.path.abspath(filepath)}_{chunk_idx}"
    # Generate a hash and convert to integer (first 8 bytes)
    return int(hashlib.md5(unique_str.encode()).hexdigest()[:8], 16) % (10**8)

def index_document(filepath):
    global is_indexing, current_file, processed_files, total_files, last_error
    
    try:
        is_indexing = True
        current_file = os.path.basename(filepath)
        
        print(f"Indexing {filepath}...")
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
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
            
        try:
            embeddings = model.encode(chunks)
            if len(embeddings) == 0 or (hasattr(embeddings, 'shape') and embeddings.shape[0] == 0):
                print(f"Warning: No embeddings generated for {filepath}")
                return
        except Exception as e:
            print(f"Error generating embeddings for {filepath}: {str(e)}")
            return
        
        # Create points for Qdrant
        points = []
        for i, (emb, chunk) in enumerate(zip(embeddings, chunks)):
            point = {
                "id": get_file_id(filepath, i),
                "vector": emb.tolist(),  # Using default vector name
                "payload": {
                    "source": os.path.basename(filepath),
                    "chunk_id": i,
                    "text": chunk,
                    "full_path": filepath
                }
            }
            points.append(point)
            print(f"Created point {i+1}/{len(chunks)} with vector size {len(emb)}")
        
        # Get or create collection
        from qdrant_client.models import Distance, VectorParams
        
        # Define collection name and vector size
        collection_name = "rag_collection"  # Must match the collection name in the RAG service
        # Handle the case where embeddings might be a NumPy array
        if len(embeddings) > 0 and len(embeddings[0]) > 0:
            vector_size = len(embeddings[0])
        else:
            # Default vector size if no embeddings were generated
            vector_size = 1024
        
        # Check if collection exists, create if it doesn't
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            try:
                print(f"Creating new collection: {collection_name}")
                # Create collection with default vector configuration
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Successfully created collection: {collection_name}")
            except Exception as e:
                print(f"Warning: Could not create collection (it might already exist): {str(e)}")
        else:
            print(f"Using existing collection: {collection_name}")
        
        # Upload points to Qdrant with retries
        max_retries = 3
        chunk_size = 10  # Smaller chunks for better error isolation
        
        print(f"Starting to process {len(points)} points in chunks of {chunk_size}")
        print(f"Vector size: {vector_size}")
        print(f"Collection name: {collection_name}")
        
        for attempt in range(max_retries):
            try:
                print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
                
                # First, verify collection exists and is accessible
                try:
                    collection_info = client.get_collection(collection_name)
                    print(f"Collection info: {collection_info}")
                except Exception as e:
                    print(f"Error getting collection info: {str(e)}")
                    # If collection doesn't exist, try to create it
                    try:
                        client.recreate_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=vector_size,
                                distance=Distance.COSINE
                            )
                        )
                        print(f"Recreated collection: {collection_name}")
                    except Exception as create_err:
                        print(f"Failed to recreate collection: {str(create_err)}")
                        raise
                
                # Process points in smaller chunks
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
                        print(f"Vector length: {len(chunk_points[0]['vector'])}")
                        print("Payload keys:", chunk_points[0]['payload'].keys())
                        print("--- End of error details ---\n")
                        
                        # If it's the last attempt, re-raise the error
                        if attempt == max_retries - 1:
                            raise
                        
                        # Otherwise, continue with next chunk
                        print(f"Continuing with next chunk...")
                        continue
                
                # If we got here, all chunks were processed successfully
                processed_files += 1
                total_files = max(total_files, processed_files)
                last_error = None
                print(f"\n✅ Successfully processed {len(points)} chunks from {filepath}")
                return  # Exit the function on success
                
            except Exception as e:
                last_error = str(e)
                print(f"\n❌ Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries - 1:  # Last attempt
                    error_msg = f"Failed to index {filepath} after {max_retries} attempts: {str(e)}"
                    print(error_msg)
                    raise Exception(error_msg) from e
                
                # Wait before retry with exponential backoff
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        # This point should never be reached due to the return on success
        raise Exception("Unexpected error: Reached end of function without completing or raising an error")
            except Exception as e3:
                print(f"Error upserting points to Qdrant (after retry): {str(e3)}")
                raise
                
    except Exception as e:
        last_error = str(e)
        error_msg = f"Error processing {filepath}: {last_error}"
        print(error_msg)
        raise Exception(error_msg) from e
        
    finally:
        is_indexing = False
        current_file = None

class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.txt', '.md')):
            index_document(event.src_path)

if __name__ == "__main__":
    # Start the status server in a separate thread
    status_thread = threading.Thread(target=run_status_server, daemon=True)
    status_thread.start()
    
    # On startup, index all existing docs
    files = [f for f in os.listdir(DOCS_PATH) 
             if os.path.isfile(os.path.join(DOCS_PATH, f)) 
             and f.lower().endswith(('.txt', '.md'))]
    
    total_files = len(files)
    processed_files = 0
    
    for fname in files:
        fpath = os.path.join(DOCS_PATH, fname)
        if os.path.isfile(fpath):
            index_document(fpath)
    
    # Watch for new files
    event_handler = DocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, DOCS_PATH, recursive=True)
    observer.start()
    print(f"Watching {DOCS_PATH} for new documents...")
    print(f"Status available at http://localhost:{STATUS_PORT}/status")
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        observer.stop()
        print("\nShutting down...")
    observer.join()
