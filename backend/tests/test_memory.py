"""
Tests for the conversation memory functionality.
"""
import pytest
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
from redis.asyncio import Redis

from ..memory import ConversationMemory, MemoryManager

# Test data
TEST_SESSION_ID = "test-session-123"
TEST_MESSAGES = [
    {"role": "user", "content": "Hello", "timestamp": "2023-01-01T12:00:00"},
    {"role": "assistant", "content": "Hi there!", "timestamp": "2023-01-01T12:00:01"},
]

@pytest.fixture
def mock_redis():
    """Fixture providing a mock Redis client."""
    redis = AsyncMock(spec=Redis)
    redis.hset = AsyncMock()
    redis.hgetall = AsyncMock()
    redis.set = AsyncMock()
    redis.get = AsyncMock()
    redis.delete = AsyncMock()
    redis.expire = AsyncMock()
    redis.exists = AsyncMock(return_value=1)
    return redis

@pytest.fixture
def memory(mock_redis):
    """Fixture providing a ConversationMemory instance with a mock Redis client."""
    return ConversationMemory(redis_client=mock_redis, session_id=TEST_SESSION_ID, ttl_hours=24)

@pytest.fixture
def memory_manager(mock_redis):
    """Fixture providing a MemoryManager instance with a mock Redis client."""
    return MemoryManager(redis_client=mock_redis)

@pytest.mark.asyncio
async def test_add_message(memory, mock_redis):
    """Test adding a message to the conversation history."""
    # Mock the current timestamp
    test_timestamp = datetime.utcnow().isoformat()
    
    # Call the method
    message_id = await memory.add_message("user", "Test message")
    
    # Verify the message was added to Redis
    assert message_id.startswith("msg_")
    mock_redis.hset.assert_called_once()
    
    # Verify the message content
    args, kwargs = mock_redis.hset.call_args
    assert args[0] == f"memory:session:{TEST_SESSION_ID}:messages"
    assert args[1].startswith("msg_")
    
    # Parse the stored message
    stored_message = json.loads(args[2])
    assert stored_message["role"] == "user"
    assert stored_message["content"] == "Test message"
    assert "timestamp" in stored_message
    
    # Verify TTL was updated
    mock_redis.expire.assert_called()

@pytest.mark.asyncio
async def test_get_recent_messages(memory, mock_redis):
    """Test retrieving recent messages from the conversation history."""
    # Mock the Redis response
    mock_messages = {
        "msg_1": json.dumps(TEST_MESSAGES[0]),
        "msg_2": json.dumps(TEST_MESSAGES[1]),
    }
    mock_redis.hgetall.return_value = mock_messages
    
    # Call the method
    messages = await memory.get_recent_messages(limit=2)
    
    # Verify the response
    assert len(messages) == 2
    assert messages[0]["content"] == "Hi there!"  # Most recent first
    assert messages[1]["content"] == "Hello"
    
    # Verify Redis was called with the correct key
    mock_redis.hgetall.assert_called_once_with(f"memory:session:{TEST_SESSION_ID}:messages")

@pytest.mark.asyncio
async def test_update_and_get_summary(memory, mock_redis):
    """Test updating and retrieving the conversation summary."""
    test_summary = "This is a test summary"
    
    # Test updating the summary
    result = await memory.update_summary(test_summary)
    assert result is True
    mock_redis.set.assert_called_with(
        f"memory:session:{TEST_SESSION_ID}:summary",
        test_summary,
        ex=86400  # 24 hours in seconds
    )
    
    # Test getting the summary
    mock_redis.get.return_value = test_summary
    summary = await memory.get_summary()
    assert summary == test_summary
    mock_redis.get.assert_called_with(f"memory:session:{TEST_SESSION_ID}:summary")

@pytest.mark.asyncio
async def test_clear(memory, mock_redis):
    """Test clearing all data for a session."""
    result = await memory.clear()
    assert result is True
    
    # Verify all keys were deleted
    mock_redis.delete.assert_called_with(
        f"memory:session:{TEST_SESSION_ID}",
        f"memory:session:{TEST_SESSION_ID}:messages",
        f"memory:session:{TEST_SESSION_ID}:summary"
    )

@pytest.mark.asyncio
async def test_memory_manager_get_memory(memory_manager, mock_redis):
    """Test getting a memory instance from the manager."""
    # Get a memory instance
    memory = memory_manager.get_memory(TEST_SESSION_ID)
    
    # Verify the instance was created with the correct parameters
    assert memory.session_id == TEST_SESSION_ID
    assert memory.redis == mock_redis
    
    # Verify the same instance is returned for the same session ID
    same_memory = memory_manager.get_memory(TEST_SESSION_ID)
    assert memory is same_memory

@pytest.mark.asyncio
async def test_memory_manager_cleanup_expired(memory_manager, mock_redis):
    """Test cleaning up expired sessions from the manager's cache."""
    # Add a session to the cache
    memory = memory_manager.get_memory(TEST_SESSION_ID)
    
    # Mock Redis to return that the key doesn't exist
    mock_redis.exists.return_value = 0
    
    # Clean up expired sessions
    removed = await memory_manager.cleanup_expired()
    
    # Verify the session was removed from the cache
    assert removed == 1
    assert TEST_SESSION_ID not in memory_manager._sessions
