import os
import logging
from qdrant_client import QdrantClient
from worker import (
    client,  # Reuse the existing client
    process_document_chunks,
    index_document,
    UPLOAD_DIR,
    DOCS_PATH,
    logger
)

def reindex_collection():
    """
    Completely reindex the collection by:
    1. Deleting the existing collection
    2. Letting the worker recreate it
    3. Processing all documents from the uploads directory
    """
    collection_name = "rag_collection"
    
    try:
        # 1. Delete the existing collection if it exists
        try:
            client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection '{collection_name}'")
        except Exception as e:
            if "not found" not in str(e).lower():
                raise
            logger.info(f"Collection '{collection_name}' doesn't exist, will create new one")
        
        # 2. Process all files in the uploads directory
        processed_count = 0
        for root, _, files in os.walk(UPLOAD_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    logger.info(f"Indexing {filepath}...")
                    if index_document(filepath, collection_name):
                        processed_count += 1
                        logger.info(f"Successfully indexed {filepath}")
                    else:
                        logger.error(f"Failed to index {filepath}")
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {str(e)}")
        
        logger.info(f"Reindexing complete. Processed {processed_count} files.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to reindex collection: {str(e)}")
        return False

if __name__ == "__main__":
    reindex_collection()
