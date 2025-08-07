# Embedding Worker

A service that processes and embeds documents for the RAG (Retrieval-Augmented Generation) system with advanced semantic chunking.

## Features

- **Multi-format Support**: Process PDF, DOCX, and plain text files
- **Semantic Chunking**: Intelligent text splitting based on document structure and content
- **Format Preservation**: Maintains document structure including headings, lists, and sections
- **Smart Merging**: Combines small chunks while respecting semantic boundaries
- **Metadata Enrichment**: Includes document and chunk-level metadata for better retrieval
- **Configurable**: Adjustable chunk sizes, overlap, and processing parameters

## Semantic Chunking

The embedding worker uses a multi-level chunking strategy:

1. **Semantic Boundary Detection**: Identifies natural breaks like headings, sections, and lists
2. **Intelligent Splitting**: Splits text at semantic boundaries while respecting chunk size constraints
3. **Context Preservation**: Maintains related content together for better context understanding
4. **Metadata Enrichment**: Adds chunk type and source information to each chunk

### Chunk Types

- `heading`: Section headings (e.g., # Title, ## Subtitle)
- `paragraph`: Regular text paragraphs
- `list_item`: Items in bulleted or numbered lists
- `sentence`: Individual sentences (used when larger chunks aren't appropriate)
- `document`: Default type for unstructured content

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
   LOG_LEVEL=INFO
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
