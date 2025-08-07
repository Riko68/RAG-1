"""
Test script for document processing functionality.
"""
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_document_processing(file_path: str):
    """Test document processing for a given file."""
    try:
        logger.info(f"Testing file: {file_path}")
        
        # Initialize processor
        processor = DocumentProcessor()
        
        # Process the file
        chunks = processor.process_file(file_path)
        
        # Log results
        logger.info(f"Processed {len(chunks)} chunks from {file_path}")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            logger.info(f"Chunk {i+1} (first 100 chars): {chunk.text[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False

if __name__ == "__main__":
    # Test with sample files (update paths as needed)
    test_files = [
        "test_docs/sample.txt",
        "test_docs/sample.pdf",
        "test_docs/sample.docx"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            success = test_document_processing(file_path)
            print(f"Test {'PASSED' if success else 'FAILED'} for {file_path}")
        else:
            print(f"File not found: {file_path}")
