"""
Test script for document processing functionality with semantic chunking.
"""
import os
import sys
import logging
import unittest
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from document_processor import DocumentProcessor, DocumentChunk

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestSemanticChunking(unittest.TestCase):
    """Test cases for semantic chunking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        self.test_texts = {
            'simple': "This is a simple test. It contains multiple sentences. "
                     "Each sentence should be properly chunked.",
            'headings': "# Header 1\n\nThis is a section.\n\n## Subheader\n\nWith some content.",
            'lists': "* Item 1\n* Item 2\n* Item 3\n\n1. First\n2. Second\n3. Third",
            'long_text': (
                "This is a longer text that should be split into multiple chunks. "
                "It contains several paragraphs with different structures.\n\n"
                "## Important Section\n\n"
                "This section should be kept together as much as possible, "
                "even if it means slightly exceeding the target chunk size.\n\n"
                "* Point 1: This is an important point\n"
                "* Point 2: This is another important point\n"
                "* Point 3: And one more for good measure"
            )
        }
    
    def test_simple_chunking(self):
        """Test basic text chunking."""
        text = self.test_texts['simple']
        chunks = self.processor._chunk_text(text, {'source': 'test'})
        self.assertGreater(len(chunks), 0)
        self.assertLessEqual(len(chunks[0].text), 500)
        
        # Verify all original text is preserved
        combined = ' '.join([c.text for c in chunks])
        self.assertIn('simple test', combined)
        self.assertIn('multiple sentences', combined)
    
    def test_heading_awareness(self):
        """Test that headings create semantic boundaries."""
        text = self.test_texts['headings']
        chunks = self.processor._chunk_text(text, {'source': 'test'})
        
        # Should have at least one chunk starting with a heading
        has_heading = any(c.text.strip().startswith('#') for c in chunks)
        self.assertTrue(has_heading, "No chunks start with a heading")
    
    def test_list_preservation(self):
        """Test that list items are preserved together."""
        text = self.test_texts['lists']
        chunks = self.processor._chunk_text(text, {'source': 'test'})
        
        # Check that list items are preserved
        list_items = []
        for chunk in chunks:
            if any(line.strip().startswith(('*', '1.', '2.', '3.')) 
                  for line in chunk.text.split('\n')):
                list_items.extend([li for li in chunk.text.split('\n') if li.strip()])
        
        self.assertGreaterEqual(len(list_items), 3, "Not all list items were preserved")
    
    def test_chunk_size_constraints(self):
        """Test that chunks respect size constraints."""
        text = self.test_texts['long_text']
        chunks = self.processor._chunk_text(text, {'source': 'test'})
        
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 1000)  # max_chunk_size = 2 * chunk_size
            self.assertIn('chunk_type', chunk.metadata)
    
    def test_metadata_preservation(self):
        """Test that metadata is properly preserved in chunks."""
        test_metadata = {
            'source': 'test_source',
            'custom_field': 'custom_value'
        }
        chunks = self.processor._chunk_text(
            self.test_texts['simple'], 
            test_metadata
        )
        
        for chunk in chunks:
            self.assertEqual(chunk.metadata['source'], 'test_source')
            self.assertEqual(chunk.metadata['custom_field'], 'custom_value')
            self.assertIn('chunk_type', chunk.metadata)

def run_tests():
    """Run all tests and return results."""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestSemanticChunking)
    test_runner = unittest.TextTestRunner(verbosity=2)
    return test_runner.run(test_suite)

def test_document_processing(file_path: str):
    """Test document processing for a given file."""
    try:
        logger.info(f"Testing file: {file_path}")
        
        # Initialize processor with semantic chunking
        processor = DocumentProcessor(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Process the file
        chunks = processor.process_file(file_path)
        
        # Log results
        logger.info(f"Processed {len(chunks)} chunks from {file_path}")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            chunk_type = chunk.metadata.get('chunk_type', 'unknown')
            logger.info(f"Chunk {i+1} ({chunk_type}, {len(chunk.text)} chars): {chunk.text[:100]}...")
        
        # Run validation tests
        validate_chunks(chunks)
        
        return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return False

def validate_chunks(chunks: List[DocumentChunk]):
    """Validate that chunks meet semantic chunking requirements."""
    if not chunks:
        logger.warning("No chunks generated")
        return
    
    # Check chunk sizes
    for i, chunk in enumerate(chunks):
        # No empty chunks
        assert chunk.text.strip(), f"Empty chunk at index {i}"
        
        # Reasonable size limits
        assert len(chunk.text) <= 1000, f"Chunk {i} too large: {len(chunk.text)} chars"
        
        # Metadata should be present
        assert 'chunk_type' in chunk.metadata, f"Missing chunk_type in chunk {i}"
        assert 'source' in chunk.metadata, f"Missing source in chunk {i}"
    
    # Check text coverage (basic check)
    all_text = ' '.join(c.text for c in chunks)
    assert len(all_text) > 0, "No text in any chunks"
    
    logger.info(f"Validated {len(chunks)} chunks with {len(all_text)} total characters")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test document processing with semantic chunking')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--file', type=str, help='Test with a specific file')
    args = parser.parse_args()
    
    if args.unit:
        # Run unit tests
        test_result = run_tests()
        exit(0 if test_result.wasSuccessful() else 1)
    
    # Test with sample files or specified file
    test_files = [args.file] if args.file else [
        "test_docs/sample.txt",
        "test_docs/sample.pdf",
        "test_docs/sample.docx"
    ]
    
    all_passed = True
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n{'='*50}\nTesting: {file_path}\n{'='*50}")
            success = test_document_processing(file_path)
            print(f"Test {'PASSED' if success else 'FAILED'} for {file_path}")
            if not success:
                all_passed = False
        else:
            print(f"File not found: {file_path}")
            all_passed = False
    
    exit(0 if all_passed else 1)
