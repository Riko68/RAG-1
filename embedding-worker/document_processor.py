"""
Document Processor Module

Handles loading and processing of different document types with smart chunking.
Supports PDF, DOCX, and plain text files.
"""
import os
import re
import magic
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import spacy
from pypdf import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata."""
    text: str
    metadata: Dict[str, Any]

class DocumentProcessor:
    """Processes different document types and splits them into chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.nlp = spacy.load("en_core_web_sm")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def process_file(self, filepath: str) -> List[DocumentChunk]:
        """Process a file and return its chunks with metadata.
        
        Args:
            filepath: Path to the file to process
            
        Returns:
            List of DocumentChunk objects
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        try:
            # Get file stats for metadata
            file_stats = os.stat(filepath)
            file_size = file_stats.st_size / (1024 * 1024)  # Size in MB
            
            # Determine file type
            file_type = self._detect_file_type(filepath)
            
            # Extract text based on file type
            if file_type == 'pdf':
                text = self._extract_pdf_text(filepath)
            elif file_type == 'docx':
                text = self._extract_docx_text(filepath)
            elif file_type == 'text':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            if not text.strip():
                logger.warning(f"No content extracted from {filepath}")
                return []
                
            # Add file metadata
            metadata = {
                'source': os.path.basename(filepath),
                'full_path': filepath,
                'file_type': file_type,
                'file_size_mb': round(file_size, 2),
                'last_modified': file_stats.st_mtime,
                'created': file_stats.st_ctime
            }
            
            # Clean and chunk the text with metadata
            return self._chunk_text(text, metadata)
            
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}", exc_info=True)
            raise
    
    def _extract_docx_text(self, filepath: str) -> str:
        """Extract text from a DOCX file with formatting.
        
        Args:
            filepath: Path to the DOCX file
            
        Returns:
            Extracted text as a string with preserved structure
        """
        try:
            doc = DocxDocument(filepath)
            text_parts = []
            
            # Extract paragraphs with their styles
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text.strip())
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text.strip())
                    text_parts.append(" | ".join(row_text))
            
            if not text_parts:
                logger.warning(f"No extractable text found in DOCX: {filepath}")
                return ""
                
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {filepath}: {str(e)}", exc_info=True)
            raise

    def _detect_file_type(self, filepath: str) -> str:
        """Detect the type of a file with improved MIME type checking.
        
        Args:
            filepath: Path to the file
            
        Returns:
            File type as a string (pdf, docx, text, etc.)
        """
        try:
            # First check by extension for known types
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext in ['.pdf']:
                return 'pdf'
            elif file_ext in ['.docx', '.doc']:
                return 'docx'
            elif file_ext in ['.txt', '.md', '.markdown']:
                return 'text'
                
            # Fall back to MIME type detection
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(filepath).lower()
            
            if 'pdf' in mime_type:
                return 'pdf'
            elif 'wordprocessingml' in mime_type or 'officedocument.word' in mime_type:
                return 'docx'
            elif 'text' in mime_type or filepath.lower().endswith(('.txt', '.md', '.markdown')):
                return 'text'
            else:
                raise ValueError(f"Unsupported file type: {mime_type}")
                
        except Exception as e:
            logger.error(f"Error detecting file type for {filepath}: {str(e)}")
            raise
    
    def _extract_pdf_text(self, filepath: str) -> str:
        """Extract text from a PDF file with improved formatting.
        
        Args:
            filepath: Path to the PDF file
            
        Returns:
            Extracted text as a string with preserved structure
        """
        try:
            reader = PdfReader(filepath)
            text_parts = []
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    # Extract text with layout information
                    page_text = page.extract_text()
                    
                    # Clean up the text
                    if page_text:
                        # Remove excessive whitespace but preserve paragraph breaks
                        page_text = re.sub(r'\s+', ' ', page_text).strip()
                        text_parts.append(f"[Page {page_num}]\n{page_text}")
                        
                except Exception as page_error:
                    logger.warning(f"Error processing page {page_num} of {filepath}: {str(page_error)}")
                    continue
            
            if not text_parts:
                logger.warning(f"No extractable text found in PDF: {filepath}")
                return ""
                
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filepath}: {str(e)}", exc_info=True)
            raise
    
    def _chunk_text(self, text: str, source_path: str) -> List[DocumentChunk]:
        """Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            source_path: Path to the source file
            
        Returns:
            List of DocumentChunk objects
        """
        # Basic cleaning
        text = text.strip()
        if not text:
            return []
            
        # Split into paragraphs first (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Use spacy for sentence boundary detection
            doc = self.nlp(para)
            sentences = [sent.text for sent in doc.sents]
            
            # Build chunks with complete sentences
            current_chunk = []
            current_length = 0
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                    
                sent_length = len(sent)
                
                # If adding this sentence would exceed chunk size, finalize current chunk
                if current_chunk and (current_length + sent_length > self.chunk_size):
                    chunk_text = " ".join(current_chunk)
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        metadata={
                            'source': os.path.basename(source_path),
                            'full_path': source_path,
                            'chunk_type': 'paragraph'
                        }
                    ))
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(sent)
                current_length += sent_length
            
            # Add the last chunk if there's any remaining text
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    metadata={
                        'source': os.path.basename(source_path),
                        'full_path': source_path,
                        'chunk_type': 'paragraph'
                    }
                ))
        
        return chunks
