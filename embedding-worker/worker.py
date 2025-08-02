import os
import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DOCS_PATH = os.environ.get("DOCS_PATH", "/documents")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(EMBEDDING_MODEL)

def chunk_text(text, chunk_size=500, overlap=100):
    # Simple word-based chunking
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def index_document(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)
    # Store in Qdrant, with file path metadata
    points = [{"id": f"{os.path.basename(filepath)}_{i}",
               "vector": emb.tolist(),
               "payload": {"source": filepath, "chunk_id": i, "text": chunk}}
              for i, (emb, chunk) in enumerate(zip(embeddings, chunks))]
    client.upsert(
        collection_name="docs",
        points=points
    )

class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.txt', '.md')):
            index_document(event.src_path)

if __name__ == "__main__":
    # On startup, index all existing docs
    for fname in os.listdir(DOCS_PATH):
        fpath = os.path.join(DOCS_PATH, fname)
        if os.path.isfile(fpath) and fname.endswith(('.txt', '.md')):
            index_document(fpath)
    # Watch for new files
    event_handler = DocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, DOCS_PATH, recursive=True)
    observer.start()
    print(f"Watching {DOCS_PATH} for new documents...")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
