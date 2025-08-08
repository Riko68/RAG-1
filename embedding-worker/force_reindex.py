from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(host='qdrant', port=6333)
collection_name = 'rag_collection'

# Get current collection info
collection_info = client.get_collection(collection_name)
print("Before reindexing:")
print("Vectors count:", collection_info.vectors_count)
print("Indexed vectors count:", collection_info.indexed_vectors_count)
print("Points count:", collection_info.points_count)

# Force reindex by updating the collection
try:
    # This will trigger a reindex
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfig(
            indexing_threshold=0,  # Force indexing
            memmap_threshold=0,    # Force all vectors to be in memory
            default_segment_number=1
        )
    )
    print("\nReindexing triggered. This might take a while...")
    
    # Wait a bit for reindexing to complete
    import time
    time.sleep(5)  # Give it some time to start reindexing
    
    # Check status
    while True:
        info = client.get_collection(collection_name)
        print(f"\rVectors: {info.vectors_count}, Indexed: {info.indexed_vectors_count}, Points: {info.points_count}", end="")
        if info.indexed_vectors_count == info.points_count:
            break
        time.sleep(1)
    
    print("\n\nReindexing completed successfully!")
    print("After reindexing:")
    print("Vectors count:", info.vectors_count)
    print("Indexed vectors count:", info.indexed_vectors_count)
    print("Points count:", info.points_count)

except Exception as e:
    print(f"\nError during reindexing: {str(e)}")
    print("Trying alternative reindexing method...")
    
    # Alternative method: create a new collection and reinsert all points
    try:
        # Get all points
        points = []
        next_offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=next_offset,
                with_vectors=True,
                with_payload=True
            )
            if not records:
                break
            points.extend(records)
        
        # Create new collection
        temp_name = f"{collection_name}_new"
        client.recreate_collection(
            collection_name=temp_name,
            vectors_config=collection_info.config.params.vectors,
            optimizers_config=models.OptimizersConfig(
                indexing_threshold=0,
                memmap_threshold=0
            )
        )
        
        # Insert points in batches
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(
                collection_name=temp_name,
                points=batch
            )
        
        # Swap collections
        client.delete_collection(collection_name)
        client.update_collection_alias(
            change_alias_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=temp_name,
                        alias_name=collection_name
                    )
                )
            ]
        )
        
        print("\nSuccessfully reindexed collection by recreating it!")
        
    except Exception as e2:
        print(f"Alternative reindexing also failed: {str(e2)}")
        print("Please check Qdrant logs for more details.")