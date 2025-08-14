import re
from typing import List
from bs4 import BeautifulSoup

class TableExtractor:
    def __init__(self):
        # Có thể bổ sung config nếu muốn
        self.header_keywords = [
            'loại', 'kích thước', 'độ lớn', 'đường', 'hệ số', 'biển',
            'chiều', 'bán kính', 'khoảng cách', 'rộng', 'dài', 'chiều dài', 'chiều rộng', 'chiều cao', 'đường kính', 'bề rộng',
            'bề dày', 'độ cao', 'độ rộng', 'độ dày', 'trọng lượng',
            'tải trọng', 'khối lượng', 'dung tích', 'thể tích', 'áp suất',
            'vận tốc', 'tốc độ', 'nhiệt độ', 'công suất', 'điện áp',
            'tần số', 'cường độ', 'mật độ', 'độ nghiêng', 'độ dốc',
            'màu', 'chất liệu', 'vật liệu', 'hình dạng', 'kí hiệu', 'ký hiệu',
            'tên', 'mô tả', 'thông số', 'tham số', 'giá trị','đường kính ngoài', 'đường kính trong', 'cự ly', 'cao độ',
            'chiều sâu', 'độ sâu', 'góc', 'độ cong', 'mức', 'cấp',
            'hạng', 'loại hình', 'mục', 'yếu tố'
        ]
        self.table_pattern = r'<<TABLE:\s*(.*?)\s*/TABLE>>'

    def extract_tables_to_searchable_text(self, text: str) -> List[str]:
        """Trích xuất nội dung <<TABLE: ... /TABLE>> và chuyển thành searchable text"""
        searchable_texts = []
        table_matches = re.findall(self.table_pattern, text, re.DOTALL)

        for table_html in table_matches:
            table_texts = self.parse_single_table_to_text(table_html)
            searchable_texts.extend(table_texts)

        return searchable_texts

    def parse_single_table_to_text(self, html_content: str) -> List[str]:
        """Parse một HTML table thành list các searchable text"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')

            if not table:
                return []

            rows = table.find_all('tr')
            if len(rows) < 2:
                return []

            headers = self.extract_headers(rows)
            if not headers:
                return []

            searchable_texts = []
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if not cells or self.is_header_like(cells, headers):
                    continue

                text_parts = []
                for i, cell in enumerate(cells):
                    if i >= len(headers):
                        break

                    cell_text = self.clean_text(cell.get_text())
                    if cell_text.strip():
                        header_key = self.normalize_header(headers[i])
                        text_parts.append(f"{header_key}: {cell_text}")

                if text_parts:
                    searchable_texts.append(" | ".join(text_parts))

            return searchable_texts

        except Exception as e:
            print(f"Error parsing table: {e}")
            return []

    def extract_headers(self, rows) -> List[str]:
        """Trích xuất headers từ các hàng đầu của table"""
        headers = []

        for row in rows[:3]:
            cells = row.find_all(['td', 'th'])
            row_texts = [self.clean_text(cell.get_text()) for cell in cells]
            if self.is_likely_header(row_texts):
                headers = row_texts
                break

        if not headers and rows:
            first_row_cells = rows[0].find_all(['td', 'th'])
            headers = [self.clean_text(cell.get_text()) for cell in first_row_cells]

        return [h for h in headers if h.strip()]

    def is_likely_header(self, texts: List[str]) -> bool:
        """Kiểm tra xem list text có phải header không"""
        if not texts:
            return False

        text_combined = ' '.join(texts).lower()
        keyword_count = sum(1 for keyword in self.header_keywords if keyword in text_combined)
        return keyword_count >= 1

    def is_header_like(self, cells, headers: List[str]) -> bool:
        """Kiểm tra xem một hàng có giống header không"""
        cell_texts = [self.clean_text(cell.get_text()) for cell in cells]

        if len(cell_texts) != len(headers):
            return False

        matches = sum(
            1 for i, text in enumerate(cell_texts)
            if i < len(headers) and text.strip().lower() == headers[i].strip().lower()
        )
        return matches >= len(headers) * 0.7

    def normalize_header(self, header: str) -> str:
        """Chuẩn hóa header thành key hợp lệ"""
        normalized = re.sub(r'[^\w\s]', '', header)
        normalized = re.sub(r'\s+', '_', normalized.strip())
        return normalized.lower()

    def clean_text(self, text: str) -> str:
        """Làm sạch text"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\r\n\t]', ' ', text)
        return text.strip()

    def get_searchable_texts(self, text: str) -> List[str]:
        """Trả về list searchable texts"""
        return self.extract_tables_to_searchable_text(text)

    def replace_tables_with_searchable_text(self, text: str) -> str:
        """Thay thế tất cả <<TABLE ... /TABLE>> bằng searchable text"""
        def replace_single_table(match):
            table_html = match.group(1)
            searchable_texts = self.parse_single_table_to_text(table_html)
            if searchable_texts:
                return "\n".join(searchable_texts)
            else:
                return match.group(0)

        return re.sub(self.table_pattern, replace_single_table, text, flags=re.DOTALL)
