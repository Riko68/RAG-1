# RAG Backend Service

This is the backend service for the RAG (Retrieval-Augmented Generation) application. It provides API endpoints for querying documents using a combination of vector search and large language models.

## Features

- Document retrieval using Qdrant vector database
- Text generation using Ollama LLM
- **Conversation Memory** - Maintain context across multiple interactions
- REST API with OpenAPI documentation
- Rate limiting and security headers
- Health check endpoints

## Conversation Memory

The conversation memory feature allows the system to maintain context across multiple interactions with a user. This is particularly useful for multi-turn conversations where previous context is important for generating relevant responses.

### How It Works

1. **Session Management**: Each conversation is assigned a unique session ID
2. **Message Storage**: Messages are stored in Redis with a configurable TTL
3. **Context Injection**: Previous conversation history is automatically injected into the LLM prompt
4. **Automatic Cleanup**: Old conversations are automatically removed based on TTL

### API Endpoints

#### Query with Memory

```http
POST /query
Content-Type: application/json

{
  "query": "What is the capital of France?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "use_memory": true,
  "top_k": 3
}
```

Response:
```json
{
  "answer": "The capital of France is Paris.",
  "sources": ["geography.pdf", "european_capitals.txt"],
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Clear Session

```http
POST /sessions/{session_id}/clear
```

#### Get Session History

```http
GET /sessions/{session_id}/history?limit=10
```

### Configuration

Environment variables for memory configuration:

- `REDIS_URL`: Redis connection URL (default: `redis://redis:6379`)
- `MEMORY_TTL_HOURS`: Time-to-live for conversation history in hours (default: `24`)
- `MAX_HISTORY_MESSAGES`: Maximum number of messages to include in context (default: `10`)

## Development

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Redis (for conversation memory)

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and update the values

### Running the Service

```bash
# Start all services
cd ..
docker-compose up -d

# Or run the backend directly (for development)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

Run the test suite:

```bash
pytest tests/ -v
```

## API Documentation

Once the service is running, you can access the interactive API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

[Your License Here]
