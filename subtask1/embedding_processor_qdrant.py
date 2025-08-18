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
                 qdrant_api_key: Optional[str] = None,
                 recreate_collections: bool = False):
        """
        Initialize the embedding processor with Qdrant
        Args:
            text_model_name: Vietnamese text embedding model
            qdrant_url: Qdrant server URL or ":memory:" for local
            qdrant_api_key: API key for Qdrant Cloud (optional)
            recreate_collections: If True, delete and recreate collections (clears existing data)
        """
        self.qdrant_url = qdrant_url
        
        # Initialize text embedding model (Vietnamese)
        print(f"Loading Vietnamese text embedding model: {text_model_name}...")
        self.text_model = SentenceTransformer(text_model_name)
        self.text_vector_size = self.text_model.get_sentence_embedding_dimension()
        
        # Initialize image embedding model (CLIP from transformers)
        print("Loading image embedding model (CLIP from transformers)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model_name = "tanganke/clip-vit-base-patch32_gtsrb"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        # Use standard OpenAI processor since the custom model doesn't have processor config
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Get actual image embedding size from model
        # Use hidden_size for pooler_output (768D) instead of projection_dim (512D)
        self.image_vector_size = self.clip_model.config.vision_config.hidden_size
        
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
        self._create_collections(recreate=recreate_collections)
        
    def _create_collections(self, recreate: bool = False):
        """Create Qdrant collections for text and images
        
        Args:
            recreate: If True, delete existing collections and recreate them
        """
        
        # Handle text collection
        try:
            self.client.get_collection(self.text_collection_name)
            if recreate:
                print(f"Deleting existing text collection '{self.text_collection_name}'...")
                self.client.delete_collection(self.text_collection_name)
                print(f"Creating new text collection '{self.text_collection_name}'...")
                self.client.create_collection(
                    collection_name=self.text_collection_name,
                    vectors_config=VectorParams(
                        size=self.text_vector_size,
                        distance=Distance.COSINE
                    )
                )
            else:
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
        
        # Handle image collection  
        try:
            self.client.get_collection(self.image_collection_name)
            if recreate:
                print(f"Deleting existing image collection '{self.image_collection_name}'...")
                self.client.delete_collection(self.image_collection_name)
                print(f"Creating new image collection '{self.image_collection_name}'...")
                self.client.create_collection(
                    collection_name=self.image_collection_name,
                    vectors_config=VectorParams(
                        size=self.image_vector_size,
                        distance=Distance.COSINE
                    )
                )
            else:
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
    

    
    def process_text_chunk(self, chunk: Dict[str, Any], array_index: Optional[int] = None) -> Tuple[str, List[float], Dict[str, Any]]:
        """
        Process a single text chunk for embedding
        
        Args:
            chunk: Chunk data from JSON (now supports sub-chunks)
            array_index: Index in the chunks array to ensure uniqueness
            
        Returns:
            Tuple of (chunk_id, embedding_vector, metadata)
        """
        # Generate unique ID for this chunk - use UUID format with array index for uniqueness
        chunk_index = chunk.get('chunk_index', 0)
        # Include array_index to handle duplicate (law_id, article_id, chunk_index) combinations
        if array_index is not None:
            unique_string = f"text_{chunk['law_id']}_{chunk['article_id']}_chunk{chunk_index}_idx{array_index}"
        else:
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
            
            # Image info (streamlined)
            "has_images": chunk.get('has_images', False),
            "article_images": chunk["article_images"],
            "chunk_images": chunk["chunk_images"],
            
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
            
            # Generate embedding using vision model directly 
            with torch.no_grad():
                # Get full model output
                outputs = self.clip_model.vision_model(**inputs)
                # Use pooler_output for 768D embedding (instead of get_image_features for 512D)
                embedding = outputs.pooler_output.cpu().numpy().flatten()
                # Normalize embedding
                #embedding = embedding / np.linalg.norm(embedding)
                #embedding = embedding.tolist()
            
            # Prepare metadata (Qdrant payload) - Optimized for RAG search
            metadata = {
                # Core identification (essential for matching)
                "law_id": chunk['law_id'],
                "law_title": chunk['law_title'],
                "article_id": str(chunk['article_id']),
                "article_title": chunk['article_title'],
                "chunk_type": "image",
                
                # Image identification
                "image_name": Path(image_path).name,
                "image_path": image_path,  # Keep for reference/debugging
                
                # Article context (helps with text-image correlation)
                "article_text_preview": chunk.get('text', '')[:200] if chunk.get('text') else '',
                
                # Image metadata for context
                "image_index_in_article": chunk.get('article_images', []).index(image_path) if image_path in chunk.get('article_images', []) else -1,
                "total_images_in_article": len(chunk.get('article_images', [])),
                
                # Note: OCR text not available in knowledge base chunks
                # Note: Removed redundant/problematic fields:
                # - "image_description": Always empty in current data
                # - "related_text_chunk": Format doesn't match actual text chunk UUIDs
                # - "image_size": Not useful for semantic similarity search
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
            
            # Process text chunk with array index to ensure uniqueness
            text_id, text_embedding, text_metadata = self.process_text_chunk(chunk, array_index=i)
            
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
                                print(f"   🔍 Expected image vector size: {self.image_vector_size}")
                            
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
                "model": "tanganke/clip-vit-base-patch32_gtsrb (768D pooler_output)"
            }
        }

def main():
    """Main function to run the embedding process with Qdrant"""
    
    # Initialize processor
    print("🚀 VLSP 2025 Law Document Embedding System (Qdrant + Vietnamese Models)")
    print("=" * 80)
    
    processor = LawEmbeddingProcessorQdrant(
        text_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        qdrant_url="./qdrant_db",  # File-based storage
        recreate_collections=True  # Clear existing data to avoid duplicates
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
    
 
if __name__ == "__main__":
    main()
