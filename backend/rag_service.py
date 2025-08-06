from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging
import os

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(
        self,
        qdrant_url: str,
        ollama_url: str,
        collection_name: str = "rag_collection",
        model_name: str = "BAAI/bge-m3",
        llm_model: str = None
    ):
        """
        Initialize the RAG service with Qdrant and Ollama.
        
        Args:
            qdrant_url: URL of the Qdrant server
            ollama_url: URL of the Ollama server
            collection_name: Name of the Qdrant collection
            model_name: Name of the embedding model
            llm_model: Name of the Ollama model to use
        """
        self.qdrant_url = qdrant_url
        self.ollama_url = ollama_url
        self.collection_name = collection_name
        self.model_name = model_name
        # Get model from environment variable or use default
        self.llm_model = llm_model or os.getenv('OLLAMA_MODEL', 'mixtral:8x7b')
        logger.info(f"Initializing RAGService with model: {self.llm_model}")
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize LLM
        logger.info(f"Initializing Ollama with model: {self.llm_model}")
        self.llm = Ollama(
            base_url=ollama_url,
            model=self.llm_model,  # Use the instance variable
            temperature=0.1,
            num_ctx=4096,  # Context window size
            timeout=60.0  # Increase timeout for larger models
        )
        
        # Define the prompt template
        self.prompt_template = """You are a helpful AI assistant. Use the following context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # Initialize the LLM chain
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
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
            
            # Format the context into a single string
            context_str = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in context])
            logger.info(f"Formatted context length: {len(context_str)} characters")
            
            # Prepare the prompt
            prompt = self.prompt.format(
                context=context_str[:500] + "..." if len(context_str) > 500 else context_str,
                question=query
            )
            logger.info(f"Prompt (truncated): {prompt[:500]}...")
            
            # Generate response using the LLM chain with timeout
            logger.info("Sending request to LLM...")
            try:
                response = await asyncio.wait_for(
                    self.llm_chain.arun({
                        "context": context_str,
                        "question": query
                    }),
                    timeout=60.0  # 60 second timeout for LLM
                )
                logger.info("Successfully received response from LLM")
                return response.strip()
                
            except asyncio.TimeoutError:
                error_msg = "LLM response timed out after 60 seconds"
                logger.error(error_msg)
                return "I'm sorry, the request took too long to process. Please try again with a more specific question."
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"I'm sorry, I encountered an error while processing your request: {str(e)}"
    
    async def ask(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Process a query using RAG.
        
        Args:
            query: The user's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing the answer and sources
        """
        logger.info(f"Processing query in ask(): {query[:100]}...")
        
        try:
            # Get relevant context
            logger.info(f"Retrieving top {top_k} relevant context chunks...")
            context = await self.get_relevant_context(query, top_k)
            
            if not context:
                logger.warning("No relevant context found for query")
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": []
                }
            
            logger.info(f"Found {len(context)} relevant context chunks")
            
            # Generate response
            logger.info("Generating response...")
            answer = await self.generate_response(query, context)
            
            # Extract unique sources
            sources = list(set([c["source"] for c in context if c.get("source")]))
            logger.info(f"Response generated with {len(sources)} sources")
            
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
