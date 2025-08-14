import re
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    sentences_count: int
    has_images: bool = False
    images_in_chunk: List[str] = None

class VietnameseTextChunker:
    """
    Smart text chunker for Vietnamese legal documents
    """
    
    def __init__(self, 
                 max_chunk_size: int = 512,  # Maximum characters per chunk
                 min_chunk_size: int = 100,  # Minimum characters per chunk
                 overlap_size: int = 50,     # Overlap between chunks
                 sentence_boundary: bool = True):  # Keep sentence boundaries
        """
        Initialize Vietnamese text chunker
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk  
            overlap_size: Number of characters to overlap between chunks
            sentence_boundary: Whether to preserve sentence boundaries
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.sentence_boundary = sentence_boundary
        
        # Vietnamese sentence ending patterns
        self.sentence_endings = r'[.!?;]\s*'
        self.strong_endings = r'[.!?]\s*'
        
        # Patterns to identify list items, subsections
        self.list_patterns = [
            r'^\d+\.\d+\.\d+\.',  # 1.2.3.
            r'^\d+\.\d+\.',       # 1.2.
            r'^\d+\.',            # 1.
            r'^[a-z]\)',          # a)
            r'^[A-Z]\)',          # A)
            r'^-\s',              # - item
            r'^\+\s',             # + item
            r'^\*\s',             # * item
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean text while preserving structure"""
        # Remove image markers but keep track of them
        text = re.sub(r'<<IMAGE:.*?/IMAGE>>', '[IMAGE]', text)
        
        # Clean up extra whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
        text = text.strip()
        
        return text
    
    def extract_image_markers(self, text: str) -> Tuple[str, List[str]]:
        """Extract image markers and return cleaned text with image list"""
        images = []
        
        # Find all image markers
        image_pattern = r'<<IMAGE:\s*([^/]+?)\s*/IMAGE>>'
        matches = re.finditer(image_pattern, text)
        
        for match in matches:
            image_name = match.group(1).strip()
            images.append(image_name)
        
        # Replace with placeholder
        cleaned_text = re.sub(image_pattern, '[IMAGE]', text)
        
        return cleaned_text, images
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while being careful with Vietnamese patterns"""
        # Split by sentence endings
        sentences = re.split(self.sentence_endings, text)
        
        # Clean up and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Rejoin very short fragments with previous sentence
        merged_sentences = []
        for i, sentence in enumerate(sentences):
            if len(sentence) < 20 and merged_sentences:
                # Very short, likely incomplete - merge with previous
                merged_sentences[-1] += '. ' + sentence
            else:
                merged_sentences.append(sentence)
        
        return merged_sentences
    
    def is_list_item_start(self, text: str) -> bool:
        """Check if text starts with a list item pattern"""
        text_start = text.lstrip()[:20]  # Check first 20 chars
        
        for pattern in self.list_patterns:
            if re.match(pattern, text_start):
                return True
        return False
    
    def create_chunks_by_sentences(self, text: str, images: List[str] = None) -> List[TextChunk]:
        """Create chunks by grouping sentences"""
        if images is None:
            images = []
            
        sentences = self.split_into_sentences(text)
        chunks = []
        
        current_chunk = ""
        current_sentences = 0
        chunk_start = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed max size
            potential_chunk = current_chunk + (' ' if current_chunk else '') + sentence
            
            # Decide whether to start new chunk
            should_split = (
                len(potential_chunk) > self.max_chunk_size and 
                len(current_chunk) >= self.min_chunk_size
            )
            
            # For list items, try to keep them together
            if (should_split and 
                self.is_list_item_start(sentence) and 
                not self.is_list_item_start(sentences[i-1] if i > 0 else "")):
                should_split = False  # Don't split before a list starts
            
            if should_split:
                # Create chunk from current content
                if current_chunk:
                    chunk_end = chunk_start + len(current_chunk)
                    
                    chunk = TextChunk(
                        text=current_chunk.strip(),
                        chunk_index=chunk_index,
                        start_char=chunk_start,
                        end_char=chunk_end,
                        sentences_count=current_sentences,
                        has_images='[IMAGE]' in current_chunk,
                        images_in_chunk=images if '[IMAGE]' in current_chunk else []
                    )
                    chunks.append(chunk)
                    
                    chunk_index += 1
                
                # Start new chunk with overlap
                if self.overlap_size > 0 and chunks:
                    # Get last few words for overlap
                    words = current_chunk.split()
                    overlap_words = words[-min(10, len(words)):]  # Last 10 words max
                    overlap_text = ' '.join(overlap_words)
                    
                    if len(overlap_text) <= self.overlap_size:
                        current_chunk = overlap_text + ' ' + sentence
                        chunk_start = chunk_end - len(overlap_text)
                    else:
                        current_chunk = sentence
                        chunk_start = chunk_end
                else:
                    current_chunk = sentence
                    chunk_start = chunk_end if chunks else 0
                
                current_sentences = 1
            else:
                # Add sentence to current chunk
                current_chunk = potential_chunk
                current_sentences += 1
        
        # Add final chunk
        if current_chunk.strip():
            chunk_end = chunk_start + len(current_chunk)
            
            chunk = TextChunk(
                text=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=chunk_start,
                end_char=chunk_end,
                sentences_count=current_sentences,
                has_images='[IMAGE]' in current_chunk,
                images_in_chunk=images if '[IMAGE]' in current_chunk else []
            )
            chunks.append(chunk)
        
        return chunks
    
    def create_chunks_sliding_window(self, text: str, images: List[str] = None) -> List[TextChunk]:
        """Create chunks using sliding window approach"""
        if images is None:
            images = []
            
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            
            # If we're not at the end, try to end at sentence boundary
            if end < len(text) and self.sentence_boundary:
                # Look for sentence ending within last 100 chars
                search_start = max(start + self.min_chunk_size, end - 100)
                sentence_end = None
                
                for match in re.finditer(self.strong_endings, text[search_start:end]):
                    sentence_end = search_start + match.end()
                
                if sentence_end:
                    end = sentence_end
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size or start + self.max_chunk_size >= len(text):
                # Count sentences in chunk
                sentences_count = len(re.findall(self.sentence_endings, chunk_text)) + 1
                
                chunk = TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    sentences_count=sentences_count,
                    has_images='[IMAGE]' in chunk_text,
                    images_in_chunk=images if '[IMAGE]' in chunk_text else []
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + self.min_chunk_size, end - self.overlap_size)
            
            # Prevent infinite loop
            if start >= end:
                start = end
        
        return chunks
    
    def chunk_article_text(self, article_text: str, method: str = "sentences") -> List[TextChunk]:
        """
        Main method to chunk article text
        
        Args:
            article_text: The article text to chunk
            method: "sentences" or "sliding" 
            
        Returns:
            List of TextChunk objects
        """
        # Extract images first
        cleaned_text, images = self.extract_image_markers(article_text)
        
        # Further clean the text
        cleaned_text = self.clean_text(cleaned_text)
        
        # Skip chunking if text is too short
        if len(cleaned_text) <= self.max_chunk_size:
            return [TextChunk(
                text=cleaned_text,
                chunk_index=0,
                start_char=0,
                end_char=len(cleaned_text),
                sentences_count=len(re.findall(self.sentence_endings, cleaned_text)) + 1,
                has_images=bool(images),
                images_in_chunk=images
            )]
        
        # Choose chunking method
        if method == "sentences":
            return self.create_chunks_by_sentences(cleaned_text, images)
        elif method == "sliding":
            return self.create_chunks_sliding_window(cleaned_text, images)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    def chunk_to_dict(self, chunk: TextChunk, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert TextChunk to dictionary format compatible with existing system"""
        return {
            "law_id": article_data["law_id"],
            "law_title": article_data["law_title"],
            "article_id": article_data["article_id"],
            "article_title": article_data["article_title"],
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "sentences_count": chunk.sentences_count,
            "has_images": chunk.has_images,
            "article_images": article_data.get("article_images", []),
            "chunk_images": chunk.images_in_chunk or [],
            "original_text_length": len(article_data.get("text", "")),
            "chunk_type": "text_chunk"
        }

def test_chunker():
    """Test the chunker with sample Vietnamese legal text"""
    
    sample_text = """
    4.1. Khi đồng thời có, bố trí các hình thức báo hiệu có ý nghĩa khác nhau cùng ở một khu vực, người tham gia giao thông đường bộ phải chấp hành báo hiệu đường bộ theo thứ tự ưu tiên từ trên xuống dưới như sau:
    4.1.1. Hiệu lệnh của người điều khiển giao thông;
    4.1.2. Tín hiệu đèn giao thông;
    4.1.3. Biển báo hiệu đường bộ;
    4.1.4. Vạch kẻ đường và các dấu hiệu khác trên mặt đường;
    4.1.5. Cọc tiêu, tường bảo vệ, rào chắn, đinh phản quang, tiêu phản quang, cột Km, cọc H;
    4.1.6. Thiết bị âm thanh báo hiệu đường bộ.
    <<IMAGE: image001.png /IMAGE>>
    4.2. Khi ở một vị trí đã có biển báo hiệu đặt cố định lại có biển báo hiệu khác đặt có tính chất tạm thời mà hai biển có ý nghĩa khác nhau thì người tham gia giao thông phải chấp hành hiệu lệnh của biển báo hiệu có tính chất tạm thời.
    """
    
    chunker = VietnameseTextChunker(
        max_chunk_size=300,
        min_chunk_size=100,
        overlap_size=50
    )
    
    print("🧪 Testing Vietnamese Text Chunker")
    print("=" * 50)
    print(f"Original text length: {len(sample_text)} characters")
    print()
    
    # Test sentence-based chunking
    chunks = chunker.chunk_article_text(sample_text, method="sentences")
    
    print(f"Created {len(chunks)} chunks:")
    print()
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  Length: {len(chunk.text)} chars")
        print(f"  Sentences: {chunk.sentences_count}")
        print(f"  Has images: {chunk.has_images}")
        print(f"  Range: {chunk.start_char}-{chunk.end_char}")
        print(f"  Text: {chunk.text[:100]}...")
        print()

if __name__ == "__main__":
    test_chunker()
