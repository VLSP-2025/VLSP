from qdrant_client import QdrantClient
from collections import Counter

# Connect to Qdrant
client = QdrantClient(path='./qdrant_db')

# Get collection info
text_info = client.get_collection('law_text_collection')
image_info = client.get_collection('law_image_collection')

print(f"📊 Collection Info:")
print(f"   Text collection: {text_info.points_count} points")
print(f"   Image collection: {image_info.points_count} points")
print()

# Get all points to check for duplicates
print("🔍 Checking for duplicate content...")
result = client.scroll('law_text_collection', limit=10000, with_payload=True, with_vectors=False)
points = result[0]

print(f"Retrieved {len(points)} points from text collection")

# Check for duplicate content based on (law_id, article_id, chunk_index)
content_keys = []
point_ids = []

for point in points:
    payload = point.payload
    key = (
        payload.get('law_id', ''),
        payload.get('article_id', ''),
        payload.get('chunk_index', 0)
    )
    content_keys.append(key)
    point_ids.append(point.id)

# Count duplicates
key_counts = Counter(content_keys)
id_counts = Counter(point_ids)

duplicates = [k for k, count in key_counts.items() if count > 1]
duplicate_ids = [i for i, count in id_counts.items() if count > 1]

print(f"\n🔍 Duplicate Analysis:")
print(f"   Unique content keys: {len(set(content_keys))}")
print(f"   Unique point IDs: {len(set(point_ids))}")
print(f"   Duplicate content: {len(duplicates)}")
print(f"   Duplicate IDs: {len(duplicate_ids)}")

if duplicates:
    print(f"\n❌ Found {len(duplicates)} duplicate content entries:")
    for i, dup_key in enumerate(duplicates[:5]):  # Show first 5
        count = key_counts[dup_key]
        print(f"   {i+1}. {dup_key} appears {count} times")

if duplicate_ids:
    print(f"\n❌ Found {len(duplicate_ids)} duplicate IDs:")
    for i, dup_id in enumerate(duplicate_ids[:3]):  # Show first 3
        count = id_counts[dup_id]
        print(f"   {i+1}. ID {dup_id} appears {count} times")

# Check if points were added multiple times
print(f"\n💡 Analysis:")
expected_chunks = 1309
actual_points = text_info.points_count
ratio = actual_points / expected_chunks

print(f"   Expected chunks: {expected_chunks}")
print(f"   Actual points: {actual_points}")
print(f"   Ratio: {ratio:.2f}")

if ratio > 1.5:
    print("   🚨 Likely cause: Embedding processor was run multiple times without clearing DB")
elif len(duplicates) > 0:
    print("   🚨 Likely cause: UUID collision or duplicate processing")
else:
    print("   ✅ Collection looks normal")
