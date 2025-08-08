from qdrant_client import QdrantClient
from qdrant_client.http import models
import time

client = QdrantClient(host='qdrant', port=6333, timeout=60.0)
collection_name = 'rag_collection'
temp_collection = f"{collection_name}_temp_{int(time.time())}"

def get_collection_stats(collection_name: str):
    try:
        info = client.get_collection(collection_name)
        print(f"Collection: {collection_name}")
        print(f"Status: {info.status}")
        print(f"Points: {info.points_count}")
        print(f"Vectors: {info.vectors_count}")
        print(f"Indexed Vectors: {info.indexed_vectors_count}")
        return info
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return None

def main():
    print("1. Getting collection info...")
    collection_info = get_collection_stats(collection_name)
    if not collection_info:
        print("Failed to get collection info. Exiting.")
        return

    # First, try a direct count to verify
    try:
        count_result = client.count(
            collection_name=collection_name,
            exact=True
        )
        print(f"\nDirect count result: {count_result.count} points")
    except Exception as e:
        print(f"Error counting points: {e}")

    # Try a small scroll to see what we're dealing with
    print("\n2. Fetching first 10 points to inspect...")
    try:
        records, _ = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_vectors=False,
            with_payload=True
        )
        print(f"First 10 points (sample):")
        for i, point in enumerate(records[:3]):  # Show first 3 as sample
            print(f"  Point {i+1}: ID={point.id}, Payload keys: {list(point.payload.keys())}")
    except Exception as e:
        print(f"Error fetching sample points: {e}")
        return

    # Ask user if they want to continue
    proceed = input("\nDo you want to proceed with reindexing? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Operation cancelled by user.")
        return

    # Proceed with reindexing
    print("\n3. Creating new collection...")
    try:
        client.recreate_collection(
            collection_name=temp_collection,
            vectors_config=collection_info.config.params.vectors,
            optimizers_config=models.OptimizersConfig(
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
        print(f"Created new collection: {temp_collection}")
    except Exception as e:
        print(f"Error creating new collection: {e}")
        return

    # Insert points in batches with progress
    print("\n4. Reindexing points...")
    offset = None
    total_processed = 0
    batch_size = 50

    try:
        while True:
            # Fetch a batch of points
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True
            )

            if not records:
                break

            # Clean and insert batch into new collection
            if records:
                # Clean the records by removing extra fields
                clean_records = []
                for record in records:
                    # Convert to dict and remove unwanted fields
                    record_dict = record.dict(exclude={'order_value', 'shard_key'}, exclude_none=True)
                    clean_records.append(record_dict)
                
                # Use the clean records for upsert
                client.upsert(
                    collection_name=temp_collection,
                    points=clean_records,
                    wait=True
                )
                total_processed += len(records)
                print(f"\rProcessed {total_processed} points...", end="")

        print(f"\nSuccessfully processed {total_processed} points.")

        # Verify new collection
        print("\n5. Verifying new collection...")
        new_info = get_collection_stats(temp_collection)
        if not new_info:
            raise Exception("Failed to verify new collection")

        # Swap collections
        print("\n6. Swapping collections...")
        try:
            client.delete_collection(collection_name)
        except Exception as e:
            print(f"Note: Could not delete old collection: {e}")

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

        print("\n✅ Reindexing completed successfully!")
        final_info = get_collection_stats(collection_name)

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user.")
        print(f"Temporary collection '{temp_collection}' was created but not made active.")
        print("You may need to clean it up manually.")
    except Exception as e:
        print(f"\n❌ Error during reindexing: {e}")
        print(f"Temporary collection '{temp_collection}' may need to be cleaned up.")
        raise

if __name__ == "__main__":
    main()