from qdrant_client import QdrantClient
from qdrant_client.http import models
import time

client = QdrantClient(host='qdrant', port=6333, timeout=60.0)  # Increased timeout
collection_name = 'rag_collection'
temp_collection = f"{collection_name}_temp_{int(time.time())}"

print("1. Getting collection info...")
try:
    collection_info = client.get_collection(collection_name)
    print(f"Current collection: {collection_name}")
    print(f"Vectors: {collection_info.vectors_count}, Indexed: {collection_info.indexed_vectors_count}, Points: {collection_info.points_count}")
    
    print("\n2. Fetching all points...")
    all_points = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_vectors=True,
            with_payload=True
        )
        if not points:
            break
        all_points.extend(points)
        print(f"\rFetched {len(all_points)} points...", end="")
    
    if not all_points:
        print("\nNo points found in the collection!")
        exit(1)
    
    print(f"\n\n3. Creating new collection: {temp_collection}")
    vector_config = collection_info.config.params.vectors
    client.recreate_collection(
        collection_name=temp_collection,
        vectors_config=vector_config,
        optimizers_config=models.OptimizersConfig(
            deleted_threshold=0.2,
            vacuum_min_vector_number=0,  # Changed from 1000 to 0
            default_segment_number=1,
            max_segment_size=10000,
            memmap_threshold=0,
            indexing_threshold=0,
            flush_interval_sec=5,
            max_optimization_threads=0
        )
    )
    
    print("4. Inserting points in batches of 50...")
    batch_size = 50
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i:i + batch_size]
        client.upsert(
            collection_name=temp_collection,
            points=batch,
            wait=True
        )
        print(f"\rInserted {min(i + len(batch), len(all_points))}/{len(all_points)} points...", end="")
    
    print("\n\n5. Verifying new collection...")
    time.sleep(2)  # Give it a moment to process
    new_info = client.get_collection(temp_collection)
    print(f"New collection status: {new_info.status}")
    print(f"Vectors: {new_info.vectors_count}, Indexed: {new_info.indexed_vectors_count}, Points: {new_info.points_count}")
    
    print("\n6. Creating alias...")
    try:
        client.delete_collection(collection_name)
    except Exception as e:
        print(f"Could not delete old collection (may not exist): {e}")
    
    client.update_collection_alias(
        change_alias_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=temp_collection,
                    alias_name=collection_name
                )
            )
        ]
    )
    
    print("\n7. Final check...")
    final_info = client.get_collection(collection_name)
    print(f"Final collection status: {final_info.status}")
    print(f"Vectors: {final_info.vectors_count}, Indexed: {final_info.indexed_vectors_count}, Points: {final_info.points_count}")
    
    if final_info.indexed_vectors_count == final_info.points_count:
        print("\n✅ Success! Collection has been reindexed.")
    else:
        print("\n⚠️  Warning: Not all vectors are indexed. Trying to force refresh...")
        client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfig(
                deleted_threshold=0.2,
                vacuum_min_vector_number=0,
                default_segment_number=1,
                max_segment_size=10000,
                memmap_threshold=0,
                indexing_threshold=0,
                flush_interval_sec=5,
                max_optimization_threads=0
            )
        )
        time.sleep(2)
        final_check = client.get_collection(collection_name)
        print(f"After refresh - Vectors: {final_check.vectors_count}, Indexed: {final_check.indexed_vectors_count}, Points: {final_check.points_count}")

except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    print("\nAttempting to clean up...")
    try:
        client.delete_collection(temp_collection)
        print(f"Deleted temporary collection: {temp_collection}")
    except:
        pass
    raise