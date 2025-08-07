# Embedding Worker

A service that processes and indexes documents for the RAG (Retrieval-Augmented Generation) system.

## Features

- **Document Processing**:
  - Supports multiple file formats: PDF, DOCX, TXT, MD
  - Smart chunking with semantic awareness
  - Metadata extraction and preservation

- **Vector Database Integration**:
  - Indexes document chunks in Qdrant
  - Handles document updates and deletions
  - Configurable chunk size and overlap

- **API Endpoints**:
  - `/status`: Check service status
  - `/search`: Query indexed documents
  - `/index`: Trigger document indexing

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. Configure environment variables (create a `.env` file):
   ```
   QDRANT_URL=http://localhost:6333
   COLLECTION_NAME=rag_documents
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

## Usage

1. Start the worker:
   ```bash
   python worker.py
   ```

2. Index documents by placing them in the `documents` directory or using the API.

3. Query the index:
   ```bash
   curl "http://localhost:8000/search?q=your+query"
   ```

## Testing

Run the test script:
```bash
python test_document_processing.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| QDRANT_URL | http://localhost:6333 | Qdrant server URL |
| COLLECTION_NAME | rag_documents | Qdrant collection name |
| CHUNK_SIZE | 1000 | Maximum characters per chunk |
| CHUNK_OVERLAP | 200 | Characters overlap between chunks |
| LOG_LEVEL | INFO | Logging level |

## Development

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run tests:
   ```bash
   pytest tests/
   ```

## License

MIT
