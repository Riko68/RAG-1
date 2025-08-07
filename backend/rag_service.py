import asyncio
import uuid
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
import os

from backend.memory import MemoryManager, ConversationMemory

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(
        self,
        qdrant_url: str,
        ollama_url: str,
        redis_client = None,
        collection_name: str = "rag_collection",
        model_name: str = "BAAI/bge-m3",
        llm_model: str = None,
        memory_ttl_hours: int = 24,
        max_history_messages: int = 10
    ):
        """
        Initialize the RAG service with Qdrant, Ollama, and optional memory.
        
        Args:
            qdrant_url: URL of the Qdrant server
            ollama_url: URL of the Ollama server
            redis_client: Optional Redis client for conversation memory
            collection_name: Name of the Qdrant collection
            model_name: Name of the embedding model
            llm_model: Name of the Ollama model to use
            memory_ttl_hours: TTL in hours for conversation memory (if Redis is used)
            max_history_messages: Maximum number of historical messages to include in context
        """
        self.qdrant_url = qdrant_url
        self.ollama_url = ollama_url
        self.collection_name = collection_name
        self.model_name = model_name
        self.max_history_messages = max_history_messages
        
        # Get model from environment variable or use default
        self.llm_model = llm_model or os.getenv('OLLAMA_MODEL', 'mixtral')
        logger.info(f"Initializing RAGService with model: {self.llm_model}")
        if ":" in self.llm_model:
            logger.warning(f"Model name '{self.llm_model}' contains a version tag. Consider using just 'mixtral'.")
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize LLM
        logger.info(f"Initializing Ollama with model: {self.llm_model}")
        self.llm = Ollama(
            base_url=ollama_url,
            model=self.llm_model,
            temperature=0.1,
            num_ctx=4096,  # Context window size
            timeout=60.0  # Increase timeout for larger models
        )
        
        # Initialize memory management if Redis is available
        self.memory_manager = None
        if redis_client:
            self.memory_manager = MemoryManager(
                redis_client=redis_client,
                default_ttl_hours=memory_ttl_hours
            )
            logger.info("Conversation memory enabled with Redis")
        else:
            logger.warning("No Redis client provided - conversation memory will be disabled")
        
        # Define the system prompt with memory context placeholders
        self.system_prompt = """You are **LexAI**, an expert-level legal assistant specialized in Swiss law. You support licensed lawyers by providing accurate, reliable, and professional responses to legal queries, using precise legal terminology and reasoning expected in legal practice in Switzerland.

You are not a general-purpose assistant. You do not give legal advice to non-lawyers. Your responses are meant solely to assist qualified legal professionals and should be verified by a licensed attorney before use in any legal or procedural context.
You speak in french, english and german, and the references you recieve may be in any of these 3 languages.
Your knowledge covers the Swiss Civil Code, Code of Obligations, Penal Code, Federal Constitution, administrative procedures, and relevant case law. You may also reference federal and cantonal legal provisions where applicable.

This system uses a **Retrieval-Augmented Generation (RAG)** architecture. Each query is accompanied by one or more **retrieved text chunks**, which may contain excerpts from Swiss laws, regulations, legal doctrine, or official commentary.

{conversation_context}

🟡 **Important usage rules**:
1. **Always rely primarily on the retrieved context** (text chunks). Quote or paraphrase directly from them whenever possible.
2. **Cite** the relevant article, paragraph, or source when referring to retrieved content (e.g., *Art. 97 CO*).
3. If the retrieved context is ambiguous or insufficient to answer the question confidently, say so. Do not hallucinate.
4. If a question is out of scope (e.g., non-Swiss law, medical, speculative finance), politely explain that you cannot assist.
5. **Maintain context** from previous messages when relevant to the current question.

Whenever it improves clarity, structure your responses using bullet points, numbered steps, or short sections.

Use a **precise, neutral, and professional tone**, appropriate for written communication between lawyers.

Your name is **LexAI**."""

        # Define the prompt template with conversation history support
        self.prompt_template = """{system_prompt}

Context from documents:
{context}

Current question: {question}

Answer as LexAI, the Swiss legal expert assistant:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["system_prompt", "context", "question"]
        )
        
        # Template for conversation history
        self.conversation_template = """## Previous conversation (most recent first):
{history}

## Current conversation:"""
        
        # Initialize the LLM chain with the system prompt included
        self.llm_chain = LLMChain(
            llm=self.llm, 
            prompt=self.prompt,
            verbose=True  # Add verbose logging for debugging
        )
    
    async def get_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from Qdrant based on the query.
        
        Args:
            query: The user's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            logger.info(f"Generating embedding for query: {query[:100]}...")
            
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            logger.info(f"Generated embedding of length: {len(query_embedding) if query_embedding else 0}")
            
            logger.info(f"Searching Qdrant collection '{self.collection_name}' for {top_k} results")
            
            # Search Qdrant for relevant documents
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_vectors=False,
                with_payload=True
            )
            
            logger.info(f"Found {len(search_results)} results from Qdrant")
            
            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "text": hit.payload.get("text", ""),
                    "source": hit.payload.get("source", "unknown"),
                    "score": hit.score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    async def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the LLM with the given context.
        
        Args:
            query: The user's question
            context: List of relevant document chunks
            
        Returns:
            Generated response from the LLM
        """
        try:
            logger.info(f"Generating response for query: {query[:100]}...")
            
            # Format the context into a single string with source attribution
            context_entries = []
            for i, c in enumerate(context, 1):
                source = c.get('source', 'unknown')
                text = c.get('text', '')
                context_entries.append(f"--- Source {i} ({source}) ---\n{text}")
            
            context_str = "\n\n".join(context_entries)
            logger.info(f"Formatted context length: {len(context_str)} characters")
            
            # Prepare the context and question for the prompt
            context_to_use = context_str[:2000] + "..." if len(context_str) > 2000 else context_str
            
            # Create the input dictionary with all required variables
            chain_input = {
                "system_prompt": self.system_prompt,
                "context": context_to_use,
                "question": query
            }
            logger.info(f"Prompt (truncated): {self.prompt.format(**chain_input)[:500]}...")
            
            # Generate response using the LLM chain with timeout
            logger.info("Sending request to LLM...")
            try:
                # Generate the response using the LLM chain with all required variables
                response = await asyncio.wait_for(
                    self.llm_chain.arun(chain_input),
                    timeout=60.0
                )
                logger.info("Successfully received response from LLM")
                return response.strip()
                
            except asyncio.TimeoutError:
                error_msg = "LLM response timed out after 60 seconds"
                logger.error(error_msg)
                return "I'm sorry, the request took too long to process. Please try again with a more specific question."
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            error_msg = f"Error in ask(): {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "answer": f"I'm sorry, I encountered an error while processing your request: {str(e)}",
                "sources": []
            }
