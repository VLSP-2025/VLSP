import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Vector database and embedding libraries
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class RAGSystem:
    def __init__(self, 
                 text_model_name: str = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
                 qdrant_url: str = "./qdrant_db",
                 qdrant_api_key: Optional[str] = None):
        """
        Initialize RAG system with the same models used for indexing
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
        self.image_vector_size = self.clip_model.config.vision_config.hidden_size
        
        # Initialize Qdrant client
        print("Initializing Qdrant client...")
        if qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")
        elif qdrant_url.startswith("./") or qdrant_url.startswith("/"):
            # File-based storage
            self.client = QdrantClient(path=qdrant_url)
            print(f"Using file-based storage at: {os.path.abspath(qdrant_url)}")
        else:
            # Server-based storage
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Collection names (must match what was used during indexing)
        self.text_collection_name = "law_text_collection"
        self.image_collection_name = "law_image_collection"
        
        print("RAG System initialized successfully!")
    
    def encode_question(self, question: str) -> List[float]:
        """
        Encode question text using the same model used for indexing
        """
        embedding = self.text_model.encode(question, normalize_embeddings=True).tolist()
        return embedding
    
    def encode_image(self, image_path: str) -> Optional[List[float]]:
        """
        Encode image using the same model used for indexing
        """
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
                # Use pooler_output for 768D embedding (same as indexing)
                embedding = outputs.pooler_output.cpu().numpy().flatten()
                # Don't normalize here to match indexing
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def search_text_similarity(self, question_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """
        Search for similar text chunks in Qdrant
        """
        try:
            search_results = self.client.search(
                collection_name=self.text_collection_name,
                query_vector=question_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": float(result.score),
                    "payload": result.payload
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching text similarity: {e}")
            return []
    
    def search_image_similarity(self, image_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """
        Search for similar images in Qdrant
        """
        try:
            search_results = self.client.search(
                collection_name=self.image_collection_name,
                query_vector=image_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": float(result.score),
                    "payload": result.payload
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching image similarity: {e}")
            return []
    def _extract_from_similarities(self, similarities: List[Dict]) -> Tuple[List[float], List[str], List[str]]:
        """Extract scores, title_ids, law_ids from similarity results"""
        scores = [result["score"] for result in similarities]
        title_ids = [result["payload"].get("article_id", "unknown") for result in similarities]
        law_ids = [result["payload"].get("law_id", "unknown") for result in similarities]
        return scores, title_ids, law_ids
    
    def _fuse_text_image_results(self, text_similarities: List[Dict], image_similarities: List[Dict],
                                text_weight: float, image_weight: float, fusion_method: str) -> Tuple[List[float], List[str], List[str]]:
        """
        Fuse text and image similarity results using different strategies
        
        Args:
            text_similarities: Results from text search
            image_similarities: Results from image search
            text_weight: Weight for text scores
            image_weight: Weight for image scores
            fusion_method: "weighted_avg" or "max"
            
        Returns:
            Tuple of (fused_scores, title_ids, law_ids)
        """
        
        # Create a mapping of (law_id, article_id) -> scores
        text_scores = {}
        image_scores = {}
        
        # Collect text scores
        for result in text_similarities:
            payload = result["payload"]
            key = (payload.get("law_id", ""), payload.get("article_id", ""))
            text_scores[key] = result["score"]
        
        # Collect image scores
        for result in image_similarities:
            payload = result["payload"]
            key = (payload.get("law_id", ""), payload.get("article_id", ""))
            image_scores[key] = result["score"]
        
        # Get all unique keys
        all_keys = set(text_scores.keys()) | set(image_scores.keys())
        
        # Calculate fused scores
        fused_results = []
        for key in all_keys:
            law_id, article_id = key
            
            text_score = text_scores.get(key, 0.0)  # Default to 0 if not found
            image_score = image_scores.get(key, 0.0)  # Default to 0 if not found
            
            if fusion_method == "weighted_avg":
                # Weighted average
                if text_score != 0 and image_score != 0:
                    fused_score = (text_score * text_weight + image_score * image_weight)
                elif text_score == 0 and image_score != 0 :
                    fused_score = image_score 
                elif text_score != 0 and image_score == 0 :
                    fused_score = text_score
                else: 
                    fused_score = 0
            #elif fusion_method == "max":
                # Take maximum score
            #    fused_score = max(text_score * text_weight, image_score * image_weight)
            #else:
                # Default to weighted average
            #    fused_score = (text_score * text_weight + image_score * image_weight)
            
            fused_results.append({
                "score": fused_score,
                "law_id": law_id,
                "article_id": article_id,
                "text_score": text_score,
                "image_score": image_score
            })
        
        # Sort by fused score (descending)
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top results and extract data
        top_results = fused_results[:max(len(text_similarities), len(image_similarities))]
        
        final_scores = [r["score"] for r in top_results]
        final_title_ids = [r["article_id"] for r in top_results]
        final_law_ids = [r["law_id"] for r in top_results]
        
        return final_scores, final_title_ids, final_law_ids




    def process_query_data(self, query_data: Dict[str, Any], 
                          top_k_text: int = 3, 
                          top_k_image: int = 3,
                          text_weight: float = 0.3,
                          image_weight: float = 0.7,
                          fusion_method: str = "weighted_avg") -> Dict[str, Any]:
        """
        Process a single query item and return RAG results with text+image fusion
        
        Args:
            query_data: Single item from query.json
            top_k_text: Number of similar text chunks to return
            top_k_image: Number of similar images to return
            text_weight: Weight for text similarity (0.0 - 1.0)
            image_weight: Weight for image similarity (0.0 - 1.0) 
            fusion_method: How to combine text and image results
                          "weighted_avg", "max", "text_only", "image_only", "separate"
            
        Returns:
            Formatted result with similarities and fusion scores
        """
        query_id = query_data["id"]
        
        # Get question embedding from query data (already embedded)
        question_embedding = None
        if "question_embedding" in query_data:
            question_embedding = query_data["question_embedding"]
        elif "embeddings" in query_data and len(query_data["embeddings"]) > 0:
            # Sometimes it might be in embeddings field
            first_embedding = query_data["embeddings"][0]
            if "question_embedding" in first_embedding:
                question_embedding = first_embedding["question_embedding"]
        
        if question_embedding is None:
            print(f"Warning: No question embedding found for {query_id}")
            return {"id": query_id, "error": "No question embedding found"}
        
        # Search for similar text chunks
        text_similarities = self.search_text_similarity(question_embedding, top_k_text)
        
        # Process each image in the query
        image_results = []
        
        if "embeddings" in query_data:
            for embedding_data in query_data["embeddings"]:
                image_id = embedding_data.get("image_id", "unknown")
                image_embedding = embedding_data.get("image_embedding", [])
                
                if not image_embedding:
                    print(f"Warning: No image embedding found for {image_id}")
                    continue
                
                # Search for similar images
                image_similarities = self.search_image_similarity(image_embedding, top_k_image)
                
                # Apply fusion strategy
                if fusion_method == "text_only":
                    final_similarities, final_title_ids, final_law_ids = self._extract_from_similarities(text_similarities)
                elif fusion_method == "image_only":
                    final_similarities, final_title_ids, final_law_ids = self._extract_from_similarities(image_similarities)
                elif fusion_method == "separate":
                    # Return both text and image results separately
                    text_sim, text_titles, text_laws = self._extract_from_similarities(text_similarities)
                    image_sim, image_titles, image_laws = self._extract_from_similarities(image_similarities)
                    
                    image_result = {
                        "image_id": image_id,
                        "image_embedding": image_embedding[:3],
                        "question_embedding": question_embedding[:3],
                        "text_similarity": text_sim,
                        "text_title_id": text_titles,
                        "text_law_id": text_laws,
                        "image_similarity": image_sim,
                        "image_title_id": image_titles,
                        "image_law_id": image_laws,
                        "full_text_similarities": text_similarities,
                        "full_image_similarities": image_similarities
                    }
                    image_results.append(image_result)
                    continue
                else:
                    # Fusion methods: weighted_avg, max
                    final_similarities, final_title_ids, final_law_ids = self._fuse_text_image_results(
                        text_similarities, image_similarities, text_weight, image_weight, fusion_method
                    )
                
                image_result = {
                    "image_id": image_id,
                    "image_embedding": image_embedding[:3],  # Show first 3 values for brevity
                    "question_embedding": question_embedding[:3],  # Show first 3 values for brevity
                    "similarity": final_similarities,
                    "title_id": final_title_ids,
                    "law_id": final_law_ids,
                    "fusion_method": fusion_method,
                    "weights": {"text": text_weight, "image": image_weight},
                    "full_text_similarities": text_similarities,  # Include full results for reference
                    "full_image_similarities": image_similarities
                }
                
                image_results.append(image_result)
        
        return {
            "id": query_id,
            "results": image_results
        }
    
    
    
    def process_query_file(self, query_file_path: str, output_file_path: str = None, 
                          top_k_text: int = 3, top_k_image: int = 3,
                          text_weight: float = 0.3, image_weight: float = 0.7,
                          fusion_method: str = "weighted_avg"):
        """
        Process entire query.json file and return RAG results with text+image fusion
        
        Args:
            query_file_path: Path to query.json file
            output_file_path: Path to save results (optional)
            top_k_text: Number of similar text chunks to return per query
            top_k_image: Number of similar images to return per image
            text_weight: Weight for text similarity (0.0 - 1.0)
            image_weight: Weight for image similarity (0.0 - 1.0)
            fusion_method: "weighted_avg", "max", "text_only", "image_only", "separate"
        """
        print(f"Processing query file: {query_file_path}")
        print(f"Fusion method: {fusion_method}, Text weight: {text_weight}, Image weight: {image_weight}")
        
        # Load query data
        with open(query_file_path, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
        
        print(f"Found {len(query_data)} queries to process")
        
        results = []
        
        for i, query_item in enumerate(query_data):
            print(f"Processing query {i+1}/{len(query_data)}: {query_item['id']}")
            
            result = self.process_query_data(
                query_item, top_k_text, top_k_image, 
                text_weight, image_weight, fusion_method
            )
            results.append(result)
        
        # Save results if output path specified
        if output_file_path:
            print(f"Saving results to: {output_file_path}")
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Processing complete! Processed {len(results)} queries")
        return results
    
    def get_collection_info(self):
        """Get information about the collections"""
        try:
            text_info = self.client.get_collection(self.text_collection_name)
            image_info = self.client.get_collection(self.image_collection_name)
            
            print(f"📊 Collection Information:")
            print(f"   Text collection: {text_info.points_count} points")
            print(f"   Image collection: {image_info.points_count} points")
            print(f"   Vector dimensions: Text={self.text_vector_size}, Image={self.image_vector_size}")
            
            return {
                "text_points": text_info.points_count,
                "image_points": image_info.points_count
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None

def main():
    """Main function to run RAG system"""
    
    print("🚀 VLSP 2025 RAG System - Question & Image Similarity Search")
    print("=" * 80)
    
    # Initialize RAG system
    rag_system = RAGSystem(
        text_model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        qdrant_url="./qdrant_db"  # Use the same path as embedding processor
    )
    
    # Check collection status
    rag_system.get_collection_info()
    
    # Process query file
    query_file = "query.json"
    output_file = "rag_results.json"
    
    if os.path.exists(query_file):
        results = rag_system.process_query_file(
            query_file_path=query_file,
            output_file_path=output_file,
            top_k_text=3,  # Top 3 similar text chunks
            top_k_image=3,  # Top 3 similar images
            text_weight = 0.3,
            image_weight= 0.7,
            fusion_method= "weighted_avg"
        )
        
        # Show sample result
        if results:
            print(f"\n📋 Sample Result (first query):")
            sample = results[0]
            print(f"Query ID: {sample['id']}")
            if 'results' in sample and sample['results']:
                first_image = sample['results'][0]
                print(f"First Image ID: {first_image['image_id']}")
                print(f"Similarity Scores: {first_image['similarity']}")
                print(f"Related Title IDs: {first_image['title_id']}")
                print(f"Related Law IDs: {first_image['law_id']}")
    else:
        print(f"❌ Query file not found: {query_file}")
        print("Please make sure the query.json file exists in the current directory")

if __name__ == "__main__":
    main()
