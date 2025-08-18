import json
import numpy as np
from typing import List, Dict, Any

def load_public_test_questions(public_test_path: str) -> Dict[str, Dict[str, str]]:
    """Load questions from public test file"""
    with open(public_test_path, 'r', encoding='utf-8') as f:
        public_test_data = json.load(f)
    
    questions_dict = {}
    for item in public_test_data:
        questions_dict[item["id"]] = {
            "image_id": item["image_id"],
            "question": item["question"]
        }
    
    return questions_dict

def process_final_output(rag_results_path: str, public_test_path: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Xử lý kết quả cuối cùng từ RAG results theo format yêu cầu nộp bài
    
    Args:
        rag_results_path: Đường dẫn tới file rag_results.json
        public_test_path: Đường dẫn tới file public test questions
        similarity_threshold: Ngưỡng similarity để lọc kết quả
    
    Returns:
        List các kết quả cuối cùng theo format submission
    """
    # Đọc dữ liệu
    with open(rag_results_path, 'r', encoding='utf-8') as f:
        rag_data = json.load(f)
    
    questions_dict = load_public_test_questions(public_test_path)
    
    final_results = []
    
    for test_case in rag_data:
        test_id = test_case["id"]
        results = test_case["results"]
        
        print(f"\nXử lý {test_id}:")
        print(f"Số lượng results: {len(results)}")
        
        # Tính average similarity cho mỗi result
        best_result = None
        best_avg_similarity = -1
        
        for i, result in enumerate(results):
            similarities = result["similarity"]
            avg_similarity = np.mean(similarities)
            
            print(f"  Result {i+1} (image_id: {result['image_id']}): avg_similarity = {avg_similarity:.4f}")
            
            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_result = result
        
        print(f"  -> Chọn result với avg_similarity cao nhất: {best_avg_similarity:.4f}")
        
        # Lọc kết quả theo threshold
        relevant_articles = []
        if best_result:
            similarities = best_result["similarity"]
            title_ids = best_result["title_id"]  # Đây chính là article_id
            law_ids = best_result["law_id"]
            
            # Lọc những kết quả có similarity >= threshold
            for sim, title_id, law_id in zip(similarities, title_ids, law_ids):
                if sim >= similarity_threshold:
                    relevant_articles.append({
                        "law_id": law_id,
                        "article_id": title_id  # title_id chính là article_id
                    })
            
            print(f"  -> Sau khi lọc với threshold {similarity_threshold}: {len(relevant_articles)} articles")
        
        # Tạo kết quả cuối cùng theo format submission
        if test_id in questions_dict:
            final_result = {
                "id": test_id,
                "image_id": questions_dict[test_id]["image_id"],
                "question": questions_dict[test_id]["question"],
                "relevant_articles": relevant_articles
            }
            
            final_results.append(final_result)
            
            print(f"  Kết quả cuối cùng cho {test_id}:")
            print(f"    - Image ID: {final_result['image_id']}")
            print(f"    - Question: {final_result['question']}")
            print(f"    - Relevant articles: {len(relevant_articles)}")
            for article in relevant_articles:
                print(f"      + {article['law_id']} / {article['article_id']}")
        else:
            print(f"  Warning: Không tìm thấy thông tin cho {test_id} trong public test")
    
    return final_results

def save_final_output(final_results: List[Dict[str, Any]], output_path: str):
    """Lưu kết quả cuối cùng ra file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nĐã lưu kết quả cuối cùng vào: {output_path}")

def main():
    # Đường dẫn file
    rag_results_path = "rag_results.json"
    public_test_path = "../public_test/vlsp_2025_public_test_task1.json"
    output_path = "vlsp_2025_submission_task1.json"
    
    # Threshold để lọc similarity (có thể điều chỉnh)
    similarity_threshold = 0.65  # Giảm threshold để có nhiều kết quả hơn
    
    print(f"Bắt đầu xử lý kết quả cuối cùng...")
    print(f"Similarity threshold: {similarity_threshold}")
    print("=" * 70)
    
    # Xử lý kết quả
    final_results = process_final_output(rag_results_path, public_test_path, similarity_threshold)
    
    # Lưu kết quả
    save_final_output(final_results, output_path)
    
    # In thống kê tổng quan
    print("\n" + "=" * 70)
    print("THỐNG KÊ TỔNG QUAN:")
    print(f"Tổng số test cases được xử lý: {len(final_results)}")
    
    total_articles = 0
    for result in final_results:
        num_articles = len(result['relevant_articles'])
        total_articles += num_articles
        print(f"\n{result['id']}:")
        print(f"  - Image ID: {result['image_id']}")
        print(f"  - Question: {result['question'][:60]}...")
        print(f"  - Số articles liên quan: {num_articles}")
        
        for article in result['relevant_articles']:
            print(f"    + {article['law_id']} / {article['article_id']}")
    
    print(f"\nTổng cộng: {total_articles} articles được đề xuất")
    print(f"Trung bình: {total_articles/len(final_results):.2f} articles/test case")

if __name__ == "__main__":
    main()
