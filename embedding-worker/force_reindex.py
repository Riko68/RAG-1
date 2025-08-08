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
        print(f"Vector config: {info.config.params.vectors}")
        
        # Get sample points to check for vectors
        try:
            records, _ = client.scroll(
                collection_name=collection_name,
                limit=1,
                with_vectors=True,
                with_payload=True
            )
            if records:
                point = records[0]
                print("\nSample point details:")
                print(f"Has vector attribute: {hasattr(point, 'vector')}")
                print(f"Vector is not None: {point.vector is not None}")
                if hasattr(point, 'vector') and point.vector is not None:
                    print(f"Vector dimension: {len(point.vector)}")
                    print(f"Vector type: {type(point.vector)}")
                    print(f"First few vector values: {point.vector[:5]}")
        except Exception as e:
            print(f"Could not check sample vectors: {e}")
            
        return info
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return None

def verify_vectors(collection_name: str, point_id: int):
    """Verify vectors are properly stored for a specific point"""
    try:
        # Get point directly
        point = client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_vectors=True
        )[0]
        
        print(f"\nVerifying vector storage for point {point_id}:")
        print(f"Vector present: {point.vector is not None}")
        if point.vector:
            print(f"Vector dimension: {len(point.vector)}")
            print(f"First 5 values: {point.vector[:5]}")
            
        # Try a similarity search
        results = client.search(
            collection_name=collection_name,
            query_vector=point.vector,
            limit=2
        )
        print(f"\nSimilarity search results:")
        for r in results:
            print(f"ID: {r.id}, Score: {r.score}")
            
    except Exception as e:
        print(f"Error verifying vectors: {e}")

# Add this after collection creation:
def verify_collection(collection_name: str):
    """Verify collection configuration and indexing status"""
    try:
        info = client.get_collection(collection_name)
        config = info.config
        print("\nCollection verification:")
        print(f"Vector size: {config.params.vectors.size}")
        print(f"Distance: {config.params.vectors.distance}")
        print(f"On disk: {config.params.vectors.on_disk}")
        print(f"HNSW config: {config.params.vectors.hnsw_config}")
        print(f"Indexing status: {info.status}")
        
        # Try to get collection telemetry
        telemetry = client.get_collection_cluster_info(collection_name)
        print(f"\nCluster info:")
        print(f"Collection status: {telemetry.status}")
        print(f"Optimizing: {telemetry.optimizing}")
        print(f"Failing: {telemetry.failing}")
        
    except Exception as e:
        print(f"Error verifying collection: {e}")

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
        collections = client.get_collections().collections
        if any(c.name == temp_collection for c in collections):
            client.delete_collection(temp_collection)
        
        # Create new collection with the same config
        vector_config = models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE,
            on_disk=False,
            hnsw_config=models.HnswConfig(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000
            ),
            # Remove the init_from parameter as it's causing validation errors
        )
        
        client.create_collection(
            collection_name=temp_collection,
            vectors_config=vector_config,
            force_recreate=True,  # Add this to ensure clean creation
            timeout=60,
            wait=True  # Make sure collection is ready before proceeding
        )
        
        # After creation, force an optimization
        client.update_collection(
            collection_name=temp_collection,
            optimizers_config=models.OptimizersConfig(
                indexing_threshold=0,  # Force immediate indexing
                memmap_threshold=0,
                flush_interval_sec=1
            ),
            timeout=60,
            wait=True
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
    max_points = count_result.count

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
                    # Debug first point before upserting
                    if clean_records:
                        first_point = clean_records[0]
                        print("\nDebug first point:")
                        print(f"ID: {first_point['id']}")
                        print(f"Vector type: {type(first_point['vector'])}")
                        print(f"Vector length: {len(first_point['vector'])}")
                        print(f"First few vector values: {first_point['vector'][:5]}")
        
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
            # First delete the old collection if it exists
            collections = client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                client.delete_collection(collection_name)
                print(f"Deleted old collection: {collection_name}")
            
            # Create the final collection with simple config
            print("Creating final collection...")
            vector_config = collection_info.config.params.vectors or models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            )
            
            print(f"Vector config: {vector_config}")
            
            # Create collection with minimal settings
            client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_config
            )
            
            # Wait a moment for collection to be ready
            time.sleep(1)
            
            # Copy all points from temp collection to the new collection
            print("Copying points to final collection...")
            offset = None
            total_copied = 0
            max_points = 300  # Safety limit - we expect around 267 points
            batch_size = 20   # Smaller batch size for better progress tracking
            
            while total_copied < max_points:
                # Fetch a batch of points from temp collection
                try:
                    records, offset = client.scroll(
                        collection_name=temp_collection,
                        limit=batch_size,
                        offset=offset,
                        with_vectors=True,
                        with_payload=True,
                        scroll_filter=models.Filter(
                            must_not=[
                                # This helps prevent infinite loops with some Qdrant versions
                                models.FieldCondition(
                                    key="id",
                                    match=models.MatchAny(any=[0])
                                )
                            ]
                        )
                    )
                except Exception as e:
                    print(f"\nError during scroll: {e}")
                    break
                
                if not records:
                    print("\nNo more records to copy.")
                    break
                    
                # Clean the records and verify vectors
                clean_records = []
                for record in records:
                    try:
                        # Verify the record has a vector
                        if not hasattr(record, 'vector') or record.vector is None:
                            print(f"\nWarning: Record {record.id} has no vector!")
                            continue
                            
                        clean_record = {
                            'id': record.id,
                            'vector': record.vector,
                            'payload': record.payload
                        }
                        clean_records.append(clean_record)
                    except Exception as e:
                        print(f"\nError cleaning record {getattr(record, 'id', 'unknown')}: {e}")
                        continue
                
                # Insert the cleaned records with wait=True for each batch
                if clean_records:
                    try:
                        # Upsert with wait=True to ensure completion
                        client.upsert(
                            collection_name=collection_name,
                            points=clean_records,
                            wait=True  # Wait for confirmation
                        )
                        
                        total_copied += len(clean_records)
                        print(f"\rCopied {total_copied} points (batch of {len(clean_records)})...", end="", flush=True)
                        
                        # Small delay between batches
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"\nError during upsert: {e}")
                        print(f"Batch size was: {len(clean_records)}")
                        if clean_records:
                            print(f"First record ID: {clean_records[0]['id']}")
                        break
                
                # If we didn't get a full batch, we're done
                if len(records) < batch_size:
                    break
                
                # Safety check - if we've copied more than expected, something might be wrong
                if total_copied > max_points:
                    print(f"\nWarning: Exceeded expected point count ({max_points}). Stopping to prevent infinite loop.")
                    break
            
            print(f"\nSuccessfully copied {total_copied} points to {collection_name}")
            
            # Clean up the temp collection
            print(f"Cleaning up temporary collection: {temp_collection}")
            client.delete_collection(temp_collection)
            
        except Exception as e:
            print(f"Error during collection swap: {e}")
            print(f"Temporary collection '{temp_collection}' may need to be cleaned up manually.")
            raise

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

    # After creating the collection and inserting points:
    print("\nVerifying vector storage...")
    verify_vectors(collection_name, 43506)  # Use one of your known point IDs

    # Add this after collection creation
    print("\nVerifying collection configuration...")
    verify_collection(temp_collection)

if __name__ == "__main__":
    main()