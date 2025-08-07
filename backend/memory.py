"""
Conversation memory module for maintaining context across multiple interactions.
Uses Redis as the storage backend with TTL-based expiration.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Manages conversation history and context using Redis as storage.
    
    Each conversation is stored with a session ID and includes:
    - Message history
    - Conversation summary
    - Metadata
    
    All keys have a TTL for automatic cleanup.
    """
    
    def __init__(self, redis_client, session_id: str, ttl_hours: int = 24):
        """
        Initialize conversation memory.
        
        Args:
            redis_client: Redis client instance
            session_id: Unique identifier for the conversation session
            ttl_hours: Time-to-live in hours for the session data
        """
        if not redis_client:
            raise ValueError("Redis client is required")
            
        self.redis = redis_client
        self.session_id = session_id
        self.ttl = ttl_hours * 3600  # Convert hours to seconds
        self.base_key = f"memory:session:{session_id}"
        
    async def add_message(self, role: str, content: str, metadata: Optional[dict] = None) -> str:
        """
        Add a message to the conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Additional metadata about the message
            
        Returns:
            Message ID for reference
        """
        if role not in ["user", "assistant"]:
            raise ValueError("Role must be 'user' or 'assistant'")
            
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        try:
            # Add to messages hash
            message_id = f"msg_{int(datetime.utcnow().timestamp() * 1000)}"
            await self.redis.hset(
                f"{self.base_key}:messages",
                message_id,
                json.dumps(message, ensure_ascii=False)
            )
            
            # Update TTL for all keys
            await self._update_ttl()
            
            logger.debug(f"Added message to session {self.session_id}: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to add message to session {self.session_id}: {str(e)}")
            raise
    
    async def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent messages from the conversation.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries, most recent first
        """
        try:
            messages = await self.redis.hgetall(f"{self.base_key}:messages")
            
            # Sort by message ID (which contains timestamp) in reverse order
            sorted_messages = sorted(
                messages.items(),
                key=lambda x: x[0],
                reverse=True
            )[:limit]
            
            # Parse JSON strings back to dicts
            result = []
            for msg_id, msg_json in sorted_messages:
                try:
                    msg_data = json.loads(msg_json)
                    result.append(msg_data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid message format for {msg_id}")
            
            # Return in chronological order
            return result[::-1]
            
        except Exception as e:
            logger.error(f"Failed to get messages for session {self.session_id}: {str(e)}")
            return []
    
    async def update_summary(self, summary: str) -> bool:
        """
        Update the conversation summary.
        
        Args:
            summary: New summary text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.redis.set(
                f"{self.base_key}:summary",
                summary,
                ex=self.ttl
            )
            logger.debug(f"Updated summary for session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update summary for session {self.session_id}: {str(e)}")
            return False
    
    async def get_summary(self) -> Optional[str]:
        """
        Get the current conversation summary.
        
        Returns:
            Summary text or None if not found
        """
        try:
            return await self.redis.get(f"{self.base_key}:summary")
        except Exception as e:
            logger.error(f"Failed to get summary for session {self.session_id}: {str(e)}")
            return None
    
    async def clear(self) -> bool:
        """
        Clear all data for this session.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete all keys for this session
            keys = [
                self.base_key,
                f"{self.base_key}:messages",
                f"{self.base_key}:summary"
            ]
            await self.redis.delete(*keys)
            logger.info(f"Cleared all data for session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session {self.session_id}: {str(e)}")
            return False
    
    async def _update_ttl(self) -> None:
        """Update TTL for all keys in this session."""
        try:
            keys = [
                self.base_key,
                f"{self.base_key}:messages",
                f"{self.base_key}:summary"
            ]
            for key in keys:
                await self.redis.expire(key, self.ttl)
        except Exception as e:
            logger.warning(f"Failed to update TTL for session {self.session_id}: {str(e)}")


class MemoryManager:
    """
    Factory and manager for conversation memory instances.
    Provides a clean interface for creating and managing multiple conversation memories.
    """
    
    def __init__(self, redis_client, default_ttl_hours: int = 24):
        """
        Initialize the memory manager.
        
        Args:
            redis_client: Redis client instance
            default_ttl_hours: Default TTL in hours for new memories
        """
        self.redis = redis_client
        self.default_ttl = default_ttl_hours
        self._sessions = {}
    
    def get_memory(self, session_id: str, ttl_hours: Optional[int] = None) -> ConversationMemory:
        """
        Get or create a conversation memory instance.
        
        Args:
            session_id: Unique identifier for the conversation
            ttl_hours: Optional TTL override in hours
            
        Returns:
            ConversationMemory instance
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(
                redis_client=self.redis,
                session_id=session_id,
                ttl_hours=ttl_hours or self.default_ttl
            )
        return self._sessions[session_id]
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions from the internal cache.
        
        Returns:
            Number of sessions removed from cache
        """
        initial_count = len(self._sessions)
        self._sessions = {
            sid: mem for sid, mem in self._sessions.items()
            if await mem.redis.exists(f"memory:session:{sid}")
        }
        return initial_count - len(self._sessions)
