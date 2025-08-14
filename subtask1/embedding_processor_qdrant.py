import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import uuid

# Vector database and embedding libraries
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class LawEmbeddingProcessorQdrant:
    def __init__(self, 
                 text_model_name: str = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                 qdrant_url: str = ":memory:",  # ":memory:" for in-memory, or "localhost:6333" for server
                 qdrant_api_key: Optional[str] = None):
        """
        Initialize the embedding processor with Qdrant
        
        Args:
            text_model_name: Vietnamese text embedding model
            qdrant_url: Qdrant server URL or ":memory:" for local
            qdrant_api_key: API key for Qdrant Cloud (optional)
        """
        self.qdrant_url = qdrant_url
        
        # Initialize text embedding model (Vietnamese)
        print(f"Loading Vietnamese text embedding model: {text_model_name}...")
        self.text_model = SentenceTransformer(text_model_name)
        self.text_vector_size = self.text_model.get_sentence_embedding_dimension()
        
        # Initialize image embedding model (CLIP from transformers)
        print("Loading image embedding model (CLIP from transformers)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        # Get actual image embedding size from model
        self.image_vector_size = self.clip_model.config.projection_dim
        
        # Initialize Qdrant client
        print("Initializing Qdrant client...")
        if qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")  # In-memory for testing
        elif qdrant_url.startswith("./") or qdrant_url.startswith("/"):
            # File-based storage
            self.client = QdrantClient(path=qdrant_url)
            print(f"Using file-based storage at: {os.path.abspath(qdrant_url)}")
        else:
            # Server-based storage
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Collection names
        self.text_collection_name = "law_text_collection"
        self.image_collection_name = "law_image_collection"
        
        # Create collections
        self._create_collections()
        
    def _create_collections(self):
        """Create Qdrant collections for text and images"""
        
        # Create text collection
        try:
            self.client.get_collection(self.text_collection_name)
            print(f"Text collection '{self.text_collection_name}' already exists")
        except:
            print(f"Creating text collection '{self.text_collection_name}'...")
            self.client.create_collection(
                collection_name=self.text_collection_name,
                vectors_config=VectorParams(
                    size=self.text_vector_size,
                    distance=Distance.COSINE
                )
            )
        
        # Create image collection  
        try:
            self.client.get_collection(self.image_collection_name)
            print(f"Image collection '{self.image_collection_name}' already exists")
        except:
            print(f"Creating image collection '{self.image_collection_name}'...")
            self.client.create_collection(
                collection_name=self.image_collection_name,
                vectors_config=VectorParams(
                    size=self.image_vector_size,
                    distance=Distance.COSINE
                )
            )
    

    
    def process_text_chunk(self, chunk: Dict[str, Any]) -> Tuple[str, List[float], Dict[str, Any]]:
        """
        Process a single text chunk for embedding
        
        Args:
            chunk: Chunk data from JSON (now supports sub-chunks)
            
        Returns:
            Tuple of (chunk_id, embedding_vector, metadata)
        """
        # Generate unique ID for this chunk - use UUID format
        chunk_index = chunk.get('chunk_index', 0)
        # Create a deterministic but unique ID based on content
        unique_string = f"text_{chunk['law_id']}_{chunk['article_id']}_chunk{chunk_index}"
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
        
        # Text is already cleaned by text_chunker, just use directly
        clean_text = chunk['text']
        
        # Generate embedding using Vietnamese model
        embedding = self.text_model.encode(clean_text, normalize_embeddings=True).tolist()
        
        # Prepare metadata (Qdrant payload) - enhanced for chunked data
        metadata = {
            "law_id": chunk['law_id'],
            "law_title": chunk['law_title'],
            "article_id": str(chunk['article_id']),
            "article_title": chunk['article_title'],
            "chunk_type": "text_chunk",
            "chunk_index": chunk_index,
            
            # Chunk-specific info
            "start_char": chunk.get('start_char', 0),
            "end_char": chunk.get('end_char', len(clean_text)),
            "sentences_count": chunk.get('sentences_count', 1),
            
            # Image info
            "has_images": chunk.get('has_images', False),
            "chunk_has_images": len(chunk.get('chunk_images', [])) > 0,
            "article_image_count": len(chunk.get('article_images', [])),
            "chunk_image_count": len(chunk.get('chunk_images', [])),
            
            # Text content
            "text_content": clean_text,  # Store full text since chunks are now smaller
            "text_length": len(clean_text),
            "original_text_length": chunk.get('original_text_length', len(clean_text)),
            
            # For search result context
            "chunk_preview": clean_text[:200] + "..." if len(clean_text) > 200 else clean_text
        }
        
        return chunk_id, embedding, metadata
    
    def process_image_chunk(self, chunk: Dict[str, Any], image_path: str, description: str = "") -> Tuple[str, List[float], Dict[str, Any]]:
        """
        Process a single image for embedding
        
        Args:
            chunk: Parent chunk data
            image_path: Path to image file
            description: Image description extracted from text
            
        Returns:
            Tuple of (chunk_id, embedding_vector, metadata)
        """
        # Generate unique ID for this image - use UUID format
        image_name = Path(image_path).stem
        unique_string = f"image_{chunk['law_id']}_{chunk['article_id']}_{image_name}"
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process image using CLIP processor
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy().flatten()
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                embedding = embedding.tolist()
            
            # Prepare metadata (Qdrant payload)
            metadata = {
                "law_id": chunk['law_id'],
                "law_title": chunk['law_title'],
                "article_id": str(chunk['article_id']),
                "article_title": chunk['article_title'],
                "chunk_type": "image",
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "image_description": description,
                "related_text_chunk": f"text_{chunk['law_id'].replace(':', '_').replace('/', '_')}_{chunk['article_id']}",
                "image_size": f"{image.size[0]}x{image.size[1]}"
            }
            
            return chunk_id, embedding, metadata
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None, None
    
    def process_chunks_file(self, chunks_file_path: str, batch_size: int = 100):
        """
        Process entire chunks.json file and add to Qdrant
        
        Args:
            chunks_file_path: Path to chunk.json file
            batch_size: Number of points to upload in each batch
        """
        print(f"Processing chunks from {chunks_file_path}...")
        
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        text_points = []
        image_points = []
        
        # Track statistics
        total_articles = len(set(f"{chunk['law_id']}-{chunk['article_id']}" for chunk in chunks))
        chunks_processed = 0
        
        print(f"📊 Statistics:")
        print(f"   Total chunks in file: {len(chunks)}")
        print(f"   Total articles: {total_articles}")
        print(f"   Text vector size: {self.text_vector_size}")
        print(f"   Image vector size: {self.image_vector_size}")
        print()
        
        for i, chunk in enumerate(chunks):
            chunk_index = chunk.get('chunk_index', 0)
            article_info = f"{chunk['law_id']} - Article {chunk['article_id']}"
            
            if chunk_index == 0:
                print(f"Processing article {chunks_processed//max(1, len([c for c in chunks if c['law_id'] == chunk['law_id'] and c['article_id'] == chunk['article_id']]))+1}/{total_articles}: {article_info}")
            
            print(f"  └─ Chunk {chunk_index+1}: {len(chunk.get('text', ''))} chars")
            
            # Process text chunk
            text_id, text_embedding, text_metadata = self.process_text_chunk(chunk)
            
            # Debug: Check vector size
            if chunks_processed == 0:  # Only show for first chunk
                print(f"   🔍 Debug: Text embedding size = {len(text_embedding)}")
            
            text_points.append(PointStruct(
                id=text_id,
                vector=text_embedding,
                payload=text_metadata
            ))
            
            chunks_processed += 1
            
            # Process image chunks if any - only for chunk_index 0 to avoid duplicates
            if chunk.get('article_images') and chunk.get('chunk_index', 0) == 0:
                # Images are now tracked in chunk_images field from text_chunker
                chunk_images = chunk.get('chunk_images', [])
                
                for j, image_path in enumerate(chunk['article_images']):
                    # Convert relative path to absolute
                    if image_path.startswith('../'):
                        image_path = os.path.join(os.path.dirname(chunks_file_path), '..', image_path.replace('../', ''))
                    
                    # Normalize path separators
                    image_path = os.path.normpath(image_path)
                    
                    if os.path.exists(image_path):
                        # No separate descriptions needed since images are pre-processed
                        description = ""
                        image_id, image_embedding, image_metadata = self.process_image_chunk(
                            chunk, image_path, description
                        )
                        
                        if image_id is not None:
                            # Debug: Check image vector size
                            if len(image_points) == 0:  # Only show for first image
                                print(f"   🔍 Debug: Image embedding size = {len(image_embedding)}")
                            
                            image_points.append(PointStruct(
                                id=image_id,
                                vector=image_embedding,
                                payload=image_metadata
                            ))
                    else:
                        print(f"Warning: Image not found: {image_path}")
            
            # Upload in batches
            if len(text_points) >= batch_size:
                print(f"Uploading batch of {len(text_points)} text points...")
                self.client.upsert(
                    collection_name=self.text_collection_name,
                    points=text_points
                )
                text_points = []
            
            if len(image_points) >= batch_size:
                print(f"Uploading batch of {len(image_points)} image points...")
                self.client.upsert(
                    collection_name=self.image_collection_name,
                    points=image_points
                )
                image_points = []
        
        # Upload remaining points
        if text_points:
            print(f"Uploading final batch of {len(text_points)} text points...")
            self.client.upsert(
                collection_name=self.text_collection_name,
                points=text_points
            )
        
        if image_points:
            print(f"Uploading final batch of {len(image_points)} image points...")
            self.client.upsert(
                collection_name=self.image_collection_name,
                points=image_points
            )
        
        # Get collection info
        text_info = self.client.get_collection(self.text_collection_name)
        image_info = self.client.get_collection(self.image_collection_name)
        
        print(f"✅ Processing complete!")
        print(f"   Text collection: {text_info.points_count} points")
        print(f"   Image collection: {image_info.points_count} points")
        print(f"   Vector dimensions: Text={self.text_vector_size}, Image={self.image_vector_size}")
    
    def search_text(self, query: str, limit: int = 5, law_id: str = None, min_score: float = 0.0) -> List[Dict]:
        """
        Search text collection
        
        Args:
            query: Search query in Vietnamese
            limit: Number of results to return
            law_id: Optional filter by law_id
            min_score: Minimum similarity score
            
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.text_model.encode(query, normalize_embeddings=True).tolist()
        
        # Prepare filter
        must_conditions = []
        if law_id:
            must_conditions.append(models.FieldCondition(
                key="law_id",
                match=models.MatchValue(value=law_id)
            ))
        
        query_filter = models.Filter(must=must_conditions) if must_conditions else None
        
        # Search
        results = self.client.search(
            collection_name=self.text_collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=min_score
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            }
            for result in results
        ]
    
    def search_image(self, query: str, limit: int = 5, law_id: str = None, min_score: float = 0.0) -> List[Dict]:
        """
        Search image collection using text query
        
        Args:
            query: Search query in Vietnamese/English
            limit: Number of results to return
            law_id: Optional filter by law_id
            min_score: Minimum similarity score
            
        Returns:
            List of search results
        """
        # Encode text query using CLIP
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)
            query_embedding = outputs.cpu().numpy().flatten()
            # Normalize embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.tolist()
        
        # Prepare filter
        must_conditions = []
        if law_id:
            must_conditions.append(models.FieldCondition(
                key="law_id",
                match=models.MatchValue(value=law_id)
            ))
        
        query_filter = models.Filter(must=must_conditions) if must_conditions else None
        
        # Search
        results = self.client.search(
            collection_name=self.image_collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=min_score
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            }
            for result in results
        ]
    
    def hybrid_search(self, query: str, limit: int = 10, law_id: str = None, 
                     text_weight: float = 0.7, image_weight: float = 0.3) -> List[Dict]:
        """
        Perform hybrid search across both text and image collections
        
        Args:
            query: Search query
            limit: Total number of results
            law_id: Optional filter by law_id
            text_weight: Weight for text results (0-1)
            image_weight: Weight for image results (0-1)
            
        Returns:
            Combined and re-ranked results
        """
        # Search both collections
        text_limit = int(limit * text_weight * 2)  # Get more to allow for reranking
        image_limit = int(limit * image_weight * 2)
        
        text_results = self.search_text(query, limit=text_limit, law_id=law_id)
        image_results = self.search_image(query, limit=image_limit, law_id=law_id)
        
        # Combine and reweight scores
        combined_results = []
        
        for result in text_results:
            result['weighted_score'] = result['score'] * text_weight
            result['type'] = 'text'
            combined_results.append(result)
        
        for result in image_results:
            result['weighted_score'] = result['score'] * image_weight  
            result['type'] = 'image'
            combined_results.append(result)
        
        # Sort by weighted score and return top results
        combined_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        return combined_results[:limit]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collections"""
        text_info = self.client.get_collection(self.text_collection_name)
        image_info = self.client.get_collection(self.image_collection_name)
        
        return {
            "text_collection": {
                "name": self.text_collection_name,
                "points_count": text_info.points_count,
                "vector_size": self.text_vector_size,
                "model": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
            },
            "image_collection": {
                "name": self.image_collection_name,
                "points_count": image_info.points_count,
                "vector_size": self.image_vector_size,
                "model": "openai/clip-vit-base-patch32"
            }
        }

def main():
    """Main function to run the embedding process with Qdrant"""
    
    # Initialize processor
    print("🚀 VLSP 2025 Law Document Embedding System (Qdrant + Vietnamese Models)")
    print("=" * 80)
    
    processor = LawEmbeddingProcessorQdrant(
        text_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        qdrant_url="./qdrant_db"  # File-based storage
    )
    
    # Process chunks file
    chunks_file = "chunk.json"
    if os.path.exists(chunks_file):
        processor.process_chunks_file(chunks_file, batch_size=50)
    else:
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    # Show collection stats
    stats = processor.get_collection_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Text: {stats['text_collection']['points_count']} points ({stats['text_collection']['vector_size']}D)")
    print(f"   Image: {stats['image_collection']['points_count']} points ({stats['image_collection']['vector_size']}D)")
    
    # Example searches
    print("\n" + "="*80)
    print("🔍 EXAMPLE SEARCHES")
    print("=" * 80)
    
    # Vietnamese text search
    print("\n📝 Vietnamese text search: 'biển báo giao thông đường bộ'")
    text_results = processor.search_text("biển báo giao thông đường bộ", limit=3)
    for i, result in enumerate(text_results):
        payload = result['payload']
        chunk_info = f"Chunk {payload.get('chunk_index', 0)+1}" if payload.get('chunk_index') is not None else ""
        print(f"{i+1}. [{payload['law_id']} - Article {payload['article_id']} {chunk_info}] {payload['article_title']}")
        print(f"   🎯 Score: {result['score']:.4f}")
        print(f"   📄 Content: {payload.get('chunk_preview', payload.get('text_content', ''))[:150]}...")
        print(f"   📏 Chunk size: {payload.get('text_length', 0)} chars")
        print()
    
    # Image search
    print("\n🖼️ Image search: 'long môn đường bộ giao thông'")
    image_results = processor.search_image("long môn đường bộ giao thông", limit=3)
    for i, result in enumerate(image_results):
        payload = result['payload']
        print(f"{i+1}. [{payload['law_id']} - Article {payload['article_id']}] {payload['article_title']}")
        print(f"   🎯 Score: {result['score']:.4f}")
        print(f"   🖼️ Image: {payload['image_name']}")
        print(f"   📝 Description: {payload['image_description']}")
        print()
    
    # Hybrid search
    print("\n🔄 Hybrid search: 'kích thước biển báo'")
    hybrid_results = processor.hybrid_search("kích thước biển báo", limit=5)
    for i, result in enumerate(hybrid_results):
        payload = result['payload']
        chunk_info = ""
        if result['type'] == 'text' and payload.get('chunk_index') is not None:
            chunk_info = f" Chunk {payload.get('chunk_index', 0)+1}"
        
        print(f"{i+1}. [{result['type'].upper()}] [{payload['law_id']} - Article {payload['article_id']}{chunk_info}] {payload['article_title']}")
        print(f"   🎯 Weighted Score: {result['weighted_score']:.4f}")
        if result['type'] == 'text':
            content = payload.get('chunk_preview', payload.get('text_content', ''))
            print(f"   📄 Content: {content[:100]}...")
            print(f"   📏 Chunk size: {payload.get('text_length', 0)} chars")
        else:
            print(f"   🖼️ Image: {payload['image_name']} - {payload['image_description']}")
        print()

if __name__ == "__main__":
    main()
