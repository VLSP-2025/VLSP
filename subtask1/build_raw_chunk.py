import os
import json
import re
from typing import List
from bs4 import BeautifulSoup
from image_extractor import ImageExtractor
from table_extractor import TableExtractor
from text_chunker import VietnameseTextChunker

law_db_path = r'../law_db'
image_path = os.path.join(law_db_path, 'images/images.fld')
vlsp2025_law_path = os.path.join(law_db_path, 'vlsp2025_law.json')
output_json_path = r'chunk.json'

# Initialize text chunker
text_chunker = VietnameseTextChunker(
    max_chunk_size=512,    # 512 chars per chunk for better embedding
    min_chunk_size=100,    # Minimum 100 chars
    overlap_size=50,       # 50 chars overlap between chunks
    sentence_boundary=True # Preserve sentence boundaries
)

result_data = []

with open(vlsp2025_law_path, 'r', encoding='utf-8') as f:
    vlsp2025_law = json.load(f)

    for law in vlsp2025_law:
        law_id = law['id']
        law_title = law['title']

        for article in law['articles']:
            article_id = article['id']
            article_title = article['title']
            
            article_content = article['text']

            if "<<TABLE:" in article_content:
                table_extractor = TableExtractor()
                article_content = table_extractor.replace_tables_with_searchable_text(article_content)

            article_images_paths = []

            if "<<IMAGE:" in article_content:
                image_extractor = ImageExtractor()
                article_images = image_extractor.extract_images(article_content)
                
                for article_image in article_images:
                    article_images_paths.append(os.path.join(image_path, article_image))

            # Create article data structure
            article_data = {
                "law_id": law_id,
                "law_title": law_title,
                "article_id": article_id,
                "text": article_content,
                "article_title": article_title,
                "article_images": article_images_paths
            }

            # Chunk the article text into smaller pieces
            text_chunks = text_chunker.chunk_article_text(article_content, method="sentences")
            
            # Add each chunk as a separate entry
            for chunk in text_chunks:
                chunk_data = text_chunker.chunk_to_dict(chunk, article_data)
                result_data.append(chunk_data)

# Ghi ra file JSON
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, ensure_ascii=False, indent=2)

print(f" Đã tạo file JSON với chunking: {output_json_path}")
print(f" Tổng số chunks: {len(result_data)}")

# Thống kê về chunks
total_articles = len([chunk for chunk in result_data if chunk.get('chunk_index') == 0])
avg_chunks_per_article = len(result_data) / total_articles if total_articles > 0 else 0
chunks_with_images = len([chunk for chunk in result_data if chunk.get('has_images')])

print(f" Thống kê:")
print(f"   - Tổng articles: {total_articles}")
print(f"   - Trung bình chunks/article: {avg_chunks_per_article:.1f}")  
print(f"   - Chunks có hình ảnh: {chunks_with_images}")
print(f"   - Kích thước chunk: {text_chunker.min_chunk_size}-{text_chunker.max_chunk_size} ký tự")