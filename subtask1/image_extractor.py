import re
from typing import List

class ImageExtractor:
    def __init__(self):
        # Có thể bổ sung config nếu muốn
        self.image_pattern = r'<<IMAGE:\s*(.*?)\s*/IMAGE>>'

    def extract_images(self, text: str) -> List[str]:
        """Trích xuất nội dung <<IMAGE: ... /IMAGE>> và chuyển thành searchable text"""
        image_paths = []
        image_matches = re.findall(self.image_pattern, text, re.DOTALL)

        for image_sentence in image_matches:
            image_paths.append(image_sentence.strip())

        return image_paths