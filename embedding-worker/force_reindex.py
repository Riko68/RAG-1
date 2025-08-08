from qdrant_client import QdrantClient
from qdrant_client.http import models
import time
from pprint import pprint

client = QdrantClient(host='qdrant', port=6333, timeout=60.0)
collection_name = 'rag_collection'
temp_collection = f"{collection_name}_temp_{int(time.time())}"

def get_collection_stats(collection_name: str):
    try:
        info = client.get_collection(collection_name)
        print(f"\nCollection: {collection_name}")
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

    # Try a direct count to verify
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
            limit=min(10, count_result.count),
            with_vectors=False,
            with_payload=True
        )
        print(f"First {len(records)} points (sample):")
        for i, point in enumerate(records[:3]):  # Show first 3 as sample
            print(f"  Point {i+1}: ID={point.id}, Payload keys: {list(point.payload.keys())}")
    except Exception as e:
        print(f"Error fetching sample points: {e}")
        return

    # Ask user if they want to continue
    proceed = input(f"\nDo you want to proceed with reindexing {count_result.count} points? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Operation cancelled by user.")
        return

    # Proceed with reindexing
    print("\n3. Creating new collection...")
    try:
        # Delete temp collection if it exists
        collections = client.get_collections()
        if any(c.name == temp_collection for c in collections.collections):
            client.delete_collection(temp_collection)
        
        # Create new collection with the same config
        client.create_collection(
            collection_name=temp_collection,
            vectors_config=collection_info.config.params.vectors
        )
        
        # Update collection settings
        client.update_collection(
            collection_name=temp_collection,
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
    max_points = count_result.count  # Don't process more points than we counted

    try:
        while total_processed < max_points:
            # Fetch a batch of points
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=min(batch_size, max_points - total_processed),
                offset=offset,
                with_vectors=True,
                with_payload=True
            )

            if not records:
                print("\nNo more records to process.")
                break

            # Clean the records
            clean_records = []
            for record in records:
                try:
                    record_dict = {
                        'id': record.id,
                        'vector': record.vector,
                        'payload': record.payload
                    }
                    clean_records.append(record_dict)
                except Exception as e:
                    print(f"\nError processing record {getattr(record, 'id', 'unknown')}: {e}")
                    continue

            # Insert batch into new collection
            if clean_records:
                try:
                    client.upsert(
                        collection_name=temp_collection,
                        points=clean_records,
                        wait=True
                    )
                    total_processed += len(clean_records)
                    print(f"\rProcessed {total_processed}/{max_points} points...", end="", flush=True)
                except Exception as e:
                    print(f"\nError inserting batch: {e}")
                    print(f"Batch size: {len(clean_records)}")
                    print(f"First record ID: {clean_records[0]['id'] if clean_records else 'N/A'}")
                    raise

        print(f"\nSuccessfully processed {total_processed} points.")

        # Verify new collection
        print("\n5. Verifying new collection...")
        new_info = get_collection_stats(temp_collection)
        if not new_info:
            raise Exception("Failed to verify new collection")

        # Swap collections
        print("\n6. Swapping collections...")
        try:
            # First remove any existing alias
            aliases = client.get_collections()
            for alias in aliases.collections:
                if alias.name == collection_name:
                    client.update_collection_alias(
                        change_alias_operations=[
                            models.DeleteAliasOperation(
                                delete_alias=models.DeleteAlias(alias_name=collection_name)
                            )
                        ]
                    )
                    break
            
            # Then delete the old collection
            client.delete_collection(collection_name)
        except Exception as e:
            print(f"Note: Could not delete old collection: {e}")

        # Create new alias
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