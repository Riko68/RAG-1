"""
Tests for the API endpoints.
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status

from ..main import app, rag_service, redis_client
from ..memory import MemoryManager, ConversationMemory

# Test client
client = TestClient(app)

# Test data
TEST_SESSION_ID = "test-session-123"
TEST_QUERY = "What is the capital of France?"
TEST_RESPONSE = {
    "answer": "The capital of France is Paris.",
    "sources": ["document1.pdf", "document2.txt"],
    "session_id": TEST_SESSION_ID
}

@pytest.fixture
def mock_rag_service():
    """Fixture to mock the RAG service."""
    with patch('app.rag_service') as mock:
        mock.query = AsyncMock(return_value=TEST_RESPONSE)
        yield mock

@pytest.fixture
def mock_redis():
    """Fixture to mock Redis client."""
    with patch('app.redis_client') as mock:
        yield mock

@pytest.fixture
def mock_memory_manager():
    """Fixture to mock the MemoryManager."""
    with patch('app.memory_manager') as mock:
        mock.get_memory.return_value = AsyncMock(spec=ConversationMemory)
        yield mock

def test_health_check():
    """Test the health check endpoint."""
    with patch('app.rag_service') as mock_service, \
         patch('app.redis_client') as mock_redis:
        
        # Mock service and Redis responses
        mock_service.qdrant_client = MagicMock()
        mock_service.qdrant_client.get_collections.return_value = MagicMock(collections=[MagicMock(name="test")])
        mock_service.llm = MagicMock()
        mock_redis.ping.return_value = True
        
        # Make the request
        response = client.get("/health")
        
        # Verify the response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["services"]["qdrant"] == "ok"
        assert data["services"]["ollama"] == "ok"

@pytest.mark.asyncio
async def test_query_endpoint_with_session():
    """Test the query endpoint with a session ID."""
    with patch('app.rag_service.query') as mock_query:
        # Setup mock
        mock_query.return_value = TEST_RESPONSE
        
        # Make the request
        response = client.post(
            "/query",
            json={
                "query": TEST_QUERY,
                "session_id": TEST_SESSION_ID,
                "use_memory": True,
                "top_k": 3
            },
            headers={"X-Role": "user"}
        )
        
        # Verify the response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["answer"] == TEST_RESPONSE["answer"]
        assert data["sources"] == TEST_RESPONSE["sources"]
        assert data["session_id"] == TEST_SESSION_ID
        
        # Verify the service was called with the correct parameters
        mock_query.assert_awaited_once_with(
            question=TEST_QUERY,
            top_k=3,
            session_id=TEST_SESSION_ID,
            use_memory=True
        )

@pytest.mark.asyncio
async def test_query_endpoint_without_session():
    """Test the query endpoint without a session ID."""
    with patch('app.rag_service.query') as mock_query, \
         patch('uuid.uuid4') as mock_uuid:
        # Setup mocks
        new_session_id = "new-session-456"
        mock_uuid.return_value = new_session_id
        mock_query.return_value = {
            **TEST_RESPONSE,
            "session_id": new_session_id
        }
        
        # Make the request
        response = client.post(
            "/query",
            json={
                "query": TEST_QUERY,
                "use_memory": True,
                "top_k": 3
            },
            headers={"X-Role": "user"}
        )
        
        # Verify the response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == new_session_id
        
        # Verify a new session ID was generated
        mock_uuid.assert_called_once()

@pytest.mark.asyncio
async def test_clear_session():
    """Test the clear session endpoint."""
    with patch('app.memory_manager') as mock_manager:
        # Setup mock
        mock_memory = AsyncMock()
        mock_memory.clear.return_value = True
        mock_manager.get_memory.return_value = mock_memory
        
        # Make the request
        response = client.post(
            f"/sessions/{TEST_SESSION_ID}/clear",
            headers={"X-Role": "user"}
        )
        
        # Verify the response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == f"Session {TEST_SESSION_ID} cleared"
        
        # Verify the memory was cleared
        mock_memory.clear.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_session_history():
    """Test getting session history."""
    with patch('app.memory_manager') as mock_manager:
        # Setup mock
        mock_memory = AsyncMock()
        mock_memory.get_recent_messages.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        mock_manager.get_memory.return_value = mock_memory
        
        # Make the request
        response = client.get(
            f"/sessions/{TEST_SESSION_ID}/history?limit=5",
            headers={"X-Role": "user"}
        )
        
        # Verify the response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == TEST_SESSION_ID
        assert len(data["messages"]) == 2
        assert data["count"] == 2
        
        # Verify the memory was queried with the correct limit
        mock_memory.get_recent_messages.assert_awaited_once_with(5)
