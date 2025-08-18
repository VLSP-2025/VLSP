"""
QdrantDB Explorer - Script để xem và khám phá vector database
"""
import os
import json
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import pandas as pd

class QdrantExplorer:
    """Tool để explore QdrantDB"""
    
    def __init__(self, db_path: str = "./qdrant_db"):
        """
        Initialize explorer
        
        Args:
            db_path: Path to Qdrant database
        """
        self.db_path = db_path
        if os.path.exists(db_path):
            self.client = QdrantClient(path=db_path)
            print(f"✅ Connected to QdrantDB at: {os.path.abspath(db_path)}")
        else:
            print(f"❌ Database not found at: {os.path.abspath(db_path)}")
            print("   Run embedding_processor_qdrant.py first to create database")
            self.client = None
    
    def get_collections_info(self) -> Dict[str, Any]:
        """Get information about all collections"""
        if not self.client:
            return {}
        
        try:
            collections = self.client.get_collections()
            info = {}
            
            for collection in collections.collections:
                collection_name = collection.name
                collection_info = self.client.get_collection(collection_name)
                
                info[collection_name] = {
                    "name": collection_name,
                    "points_count": collection_info.points_count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance,
                    "indexed": collection_info.status,
                }
            
            return info
        except Exception as e:
            print(f"Error getting collections info: {e}")
            return {}
    
    def show_collections_summary(self):
        """Display summary of all collections"""
        print("📊 QDRANT DATABASE OVERVIEW")
        print("=" * 50)
        
        collections_info = self.get_collections_info()
        
        if not collections_info:
            print("❌ No collections found or database error")
            return
        
        total_points = sum(info["points_count"] for info in collections_info.values())
        
        print(f"📁 Database path: {os.path.abspath(self.db_path)}")
        print(f"🔢 Total collections: {len(collections_info)}")
        print(f"📄 Total points: {total_points:,}")
        print()
        
        for name, info in collections_info.items():
            print(f"📚 Collection: {name}")
            print(f"   Points: {info['points_count']:,}")
            print(f"   Vector size: {info['vector_size']}D")
            print(f"   Distance: {info['distance']}")
            print(f"   Status: {info['indexed']}")
            print()
    
    def sample_points(self, collection_name: str, limit: int = 5, with_vectors: bool = False) -> List[Dict]:
        """Get sample points from a collection"""
        if not self.client:
            return []
        
        try:
            result = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=with_vectors
            )
            
            points = []
            for point in result[0]:
                point_data = {
                    "id": point.id,
                    "payload": point.payload
                }
                if with_vectors and point.vector:
                    point_data["vector"] = point.vector
                points.append(point_data)
            
            return points
        except Exception as e:
            print(f"Error sampling points: {e}")
            return []
    
    def show_sample_data(self, collection_name: str, limit: int = 3, show_vectors: bool = False):
        """Display sample data from collection"""
        print(f"🔍 SAMPLE DATA FROM '{collection_name}'")
        print("=" * 50)
        
        samples = self.sample_points(collection_name, limit, with_vectors=show_vectors)
        
        if not samples:
            print("❌ No data found or error occurred")
            return
        
        for i, sample in enumerate(samples, 1):
            print(f"📄 Sample {i}:")
            print(f"   ID: {sample['id']}")
            
            payload = sample['payload']
            
            # Show key information based on type
            if payload.get('chunk_type') == 'text_chunk':
                print(f"   Type: Text Chunk")
                print(f"   Law: {payload.get('law_id', 'N/A')}")
                print(f"   Article: {payload.get('article_id', 'N/A')}")
                print(f"   Title: {payload.get('article_title', 'N/A')}")
                print(f"   Chunk Index: {payload.get('chunk_index', 'N/A')}")
                print(f"   Text Length: {payload.get('text_length', 'N/A')} chars")
                print(f"   Has Images: {payload.get('has_images', 'N/A')}")
                if payload.get('article_images'):
                    print(f"   Article Images: {len(payload.get('article_images', []))} total")
                if payload.get('chunk_images'):
                    print(f"   Chunk Images: {len(payload.get('chunk_images', []))} in this chunk")
                print(f"   Content Preview: {payload.get('chunk_preview', payload.get('text_content', 'N/A'))[:100]}...")
                
            elif payload.get('chunk_type') == 'image':
                print(f"   Type: Image")
                print(f"   Law: {payload.get('law_id', 'N/A')}")
                print(f"   Article: {payload.get('article_id', 'N/A')}")
                print(f"   Title: {payload.get('article_title', 'N/A')}")
                print(f"   Image: {payload.get('image_name', 'N/A')}")
                print(f"   Path: {payload.get('image_path', 'N/A')}")
                print(f"   Image Index: {payload.get('image_index_in_article', 'N/A')} / {payload.get('total_images_in_article', 'N/A')}")
                article_preview = payload.get('article_text_preview', '')
                if article_preview:
                    print(f"   Article Preview: {article_preview[:100]}...")
                else:
                    print(f"   Article Preview: N/A")
            
            # Show vector embedding if requested
            if show_vectors and 'vector' in sample:
                vector = sample['vector']
                print(f"   🎯 Vector Embedding:")
                print(f"      Dimensions: {len(vector)}")
                print(f"      First 10 values: {[round(v, 6) for v in vector[:10]]}")
                print(f"      Middle 10 values: {[round(v, 6) for v in vector[len(vector)//2-5:len(vector)//2+5]]}")
                print(f"      Last 10 values: {[round(v, 6) for v in vector[-10:]]}")
                
                # Calculate vector statistics
                import numpy as np
                v_array = np.array(vector)
                norm = np.linalg.norm(v_array)
                
                print(f"      📈 Statistics:")
                print(f"         Min: {v_array.min():.6f}")
                print(f"         Max: {v_array.max():.6f}")
                print(f"         Mean: {v_array.mean():.6f}")
                print(f"         Std: {v_array.std():.6f}")
                print(f"         L2 Norm: {norm:.6f}")
                print(f"         Normalized: {'✅ Yes' if abs(norm - 1.0) < 0.01 else '❌ No'}")
            
            print()
    
    def search_by_law(self, collection_name: str, law_id: str, limit: int = 10) -> List[Dict]:
        """Search points by law_id"""
        if not self.client:
            return []
        
        try:
            result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="law_id",
                            match=MatchValue(value=law_id)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            return [
                {
                    "id": point.id,
                    "payload": point.payload
                }
                for point in result[0]
            ]
        except Exception as e:
            print(f"Error searching by law: {e}")
            return []
    
    def show_law_summary(self, law_id: str):
        """Show summary of a specific law in database"""
        print(f"📋 LAW SUMMARY: {law_id}")
        print("=" * 50)
        
        collections_info = self.get_collections_info()
        
        for collection_name in collections_info.keys():
            results = self.search_by_law(collection_name, law_id, limit=100)
            
            if results:
                print(f"\n📚 Collection: {collection_name}")
                print(f"   Total chunks: {len(results)}")
                
                if collection_name == "law_text_collection":
                    # Group by article
                    articles = {}
                    for result in results:
                        article_id = result['payload'].get('article_id', 'unknown')
                        if article_id not in articles:
                            articles[article_id] = []
                        articles[article_id].append(result)
                    
                    print(f"   Articles: {len(articles)}")
                    
                    # Show article breakdown
                    for article_id, chunks in sorted(articles.items())[:5]:  # Show first 5
                        print(f"     Article {article_id}: {len(chunks)} chunks")
                        if chunks:
                            title = chunks[0]['payload'].get('article_title', 'No title')
                            print(f"       Title: {title}")
                
                elif collection_name == "law_image_collection":
                    # Show image info with new metadata
                    for result in results[:3]:  # Show first 3
                        payload = result['payload']
                        print(f"     Image: {payload.get('image_name', 'N/A')}")
                        print(f"       Article: {payload.get('article_id', 'N/A')} ({payload.get('article_title', 'No title')})")
                        print(f"       Position: {payload.get('image_index_in_article', 'N/A')} / {payload.get('total_images_in_article', 'N/A')}")
                        article_preview = payload.get('article_text_preview', '')
                        if article_preview:
                            print(f"       Context: {article_preview[:50]}...")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the database"""
        if not self.client:
            return {}
        
        collections_info = self.get_collections_info()
        stats = {
            "collections": collections_info,
            "total_points": sum(info["points_count"] for info in collections_info.values())
        }
        
        # Get law distribution for text collection
        if "law_text_collection" in collections_info:
            try:
                # Get all points to analyze
                result = self.client.scroll(
                    collection_name="law_text_collection",
                    limit=10000,  # Large number to get all
                    with_payload=True,
                    with_vectors=False
                )
                
                law_counts = {}
                chunk_sizes = []
                
                for point in result[0]:
                    payload = point.payload
                    law_id = payload.get('law_id', 'unknown')
                    law_counts[law_id] = law_counts.get(law_id, 0) + 1
                    
                    text_length = payload.get('text_length', 0)
                    if text_length > 0:
                        chunk_sizes.append(text_length)
                
                stats["law_distribution"] = law_counts
                stats["chunk_size_stats"] = {
                    "avg": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                    "min": min(chunk_sizes) if chunk_sizes else 0,
                    "max": max(chunk_sizes) if chunk_sizes else 0,
                    "total_chunks": len(chunk_sizes)
                }
                
            except Exception as e:
                print(f"Error getting detailed stats: {e}")
        
        return stats
    
    def analyze_metadata_fields(self, collection_name: str, limit: int = 100):
        """Analyze metadata fields in a collection"""
        if not self.client:
            print("❌ No database connection")
            return
        
        try:
            print(f"🔍 Analyzing metadata fields in {collection_name}")
            print("=" * 50)
            
            result = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Collect all unique fields
            all_fields = set()
            field_types = {}
            field_examples = {}
            
            for point in result[0]:
                payload = point.payload
                for key, value in payload.items():
                    all_fields.add(key)
                    
                    # Track field types
                    field_type = type(value).__name__
                    if key not in field_types:
                        field_types[key] = set()
                    field_types[key].add(field_type)
                    
                    # Save examples
                    if key not in field_examples:
                        field_examples[key] = []
                    if len(field_examples[key]) < 3:  # Keep max 3 examples
                        field_examples[key].append(value)
            
            # Display analysis
            print(f"📊 Found {len(all_fields)} unique metadata fields:")
            print()
            
            for field in sorted(all_fields):
                types = ", ".join(field_types[field])
                examples = field_examples[field][:2]  # Show first 2 examples
                
                print(f"🔹 {field}")
                print(f"   Type(s): {types}")
                print(f"   Examples: {examples}")
                print()
                
        except Exception as e:
            print(f"❌ Error analyzing metadata: {e}")
    
    def export_to_json(self, collection_name: str, output_file: str, limit: Optional[int] = None, with_vectors: bool = False):
        """Export collection data to JSON file"""
        if not self.client:
            print("❌ No database connection")
            return
        
        try:
            print(f"📤 Exporting {collection_name} to {output_file}...")
            if with_vectors:
                print("   ⚠️ Including vectors - file will be large!")
            
            result = self.client.scroll(
                collection_name=collection_name,
                limit=limit or 10000,  # Default large number
                with_payload=True,
                with_vectors=with_vectors
            )
            
            data = []
            for point in result[0]:
                point_data = {
                    "id": point.id,
                    "payload": point.payload
                }
                if with_vectors and point.vector:
                    point_data["vector"] = point.vector
                data.append(point_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(output_file) / 1024 / 1024  # MB
            vector_info = "with vectors" if with_vectors else "without vectors"
            print(f"✅ Exported {len(data)} points to {output_file} ({vector_info})")
            print(f"📏 File size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")

def main():
    """Main exploration interface"""
    explorer = QdrantExplorer()
    
    if not explorer.client:
        return
    
    while True:
        print("\n" + "="*60)
        print("🔍 QDRANT EXPLORER MENU")
        print("="*60)
        print("1. Show collections overview")
        print("2. Sample data from text collection")
        print("3. Sample data from image collection") 
        print("4. Sample data WITH VECTORS from text collection")
        print("5. Sample data WITH VECTORS from image collection")
        print("6. Search by law ID")
        print("7. Show database statistics")
        print("8. Analyze metadata fields")
        print("9. Export collection to JSON")
        print("0. Exit")
        print("-"*60)
        
        choice = input("Enter your choice (0-9): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            explorer.show_collections_summary()
        elif choice == "2":
            explorer.show_sample_data("law_text_collection")
        elif choice == "3":
            explorer.show_sample_data("law_image_collection")
        elif choice == "4":
            print("\n🎯 SHOWING TEXT VECTORS:")
            explorer.show_sample_data("law_text_collection", limit=2, show_vectors=True)
        elif choice == "5":
            print("\n🎯 SHOWING IMAGE VECTORS:")
            explorer.show_sample_data("law_image_collection", limit=2, show_vectors=True)
        elif choice == "6":
            law_id = input("Enter law ID (e.g., 'QCVN 41:2024/BGTVT'): ").strip()
            if law_id:
                explorer.show_law_summary(law_id)
        elif choice == "7":
            stats = explorer.get_statistics()
            print("\n📊 DATABASE STATISTICS")
            print("="*50)
            print(f"Total points: {stats.get('total_points', 0):,}")
            
            if "law_distribution" in stats:
                print(f"\nTop laws by chunk count:")
                law_dist = stats["law_distribution"]
                for law, count in sorted(law_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {law}: {count} chunks")
            
            if "chunk_size_stats" in stats:
                chunk_stats = stats["chunk_size_stats"]
                print(f"\nChunk size statistics:")
                print(f"  Average: {chunk_stats['avg']:.0f} characters")
                print(f"  Range: {chunk_stats['min']}-{chunk_stats['max']} characters")
                print(f"  Total chunks: {chunk_stats['total_chunks']}")
                
        elif choice == "8":
            print("\n🔍 Metadata Analysis:")
            collection = input("Enter collection name (law_text_collection/law_image_collection): ").strip()
            if collection:
                limit = input("Number of points to analyze (default 100): ").strip()
                limit = int(limit) if limit.isdigit() else 100
                explorer.analyze_metadata_fields(collection, limit)
        elif choice == "9":
            collection_name = input("Enter collection name (law_text_collection/law_image_collection): ").strip()
            if collection_name not in ["law_text_collection", "law_image_collection"]:
                print("❌ Invalid collection name")
                continue
                
            output_file = input("Enter output file name (e.g., 'export.json'): ").strip()
            if not output_file:
                print("❌ Invalid file name")
                continue
                
            include_vectors = input("Include vector embeddings? (y/n): ").strip().lower() == 'y'
            limit = input("Limit number of points (default: all): ").strip()
            limit = int(limit) if limit.isdigit() else None
            
            explorer.export_to_json(collection_name, output_file, limit, include_vectors)
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
