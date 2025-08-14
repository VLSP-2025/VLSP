=============README===================
1. Thư mục law_db: chứa thông tin về 2 điều luật phục vụ cho cuộc thi, gồm:
   + Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ (Số: QCVN 41:2024/BGTVT).  
   + Luật trật tự giao thông đường bộ (Số: 36/2024/QH15) 

   Dữ liệu được mã hoá theo cấu trúc json, trong đó, mỗi object sẽ có id ứng với từng điều luật (articles) của 2 luật trên.
    - Với các sample có hình ảnh, dữ liệu hình ảnh được mã hoá trong đoạn template: "<<IMAGE: tên_ảnh.jpg /IMAGE>>" 
    - Với các sample dạng bảng, dữ liệu được mã hoá trong đoạn template: <<TABLE: html_table... /TABLE>>, trong đó, bảng được trình bày theo dạng HTML. Các đội có thể dùng các công cụ như pandas read_html() để đọc và parse dữ liệu bảng về Dataframe (nếu cần).

   Thư mục images: chứa các hình ảnh liên quan đến các điều luật (đã được nén dưới dạng .zip)

2. Dữ liệu training: gồm 1 file json với cấu trúc như sau:
   + "id": id của sample.
   + "image_id": id của image. Tên image tương ứng trong thư mục "images" sẽ là "<id>.jpg"
   + "relevant_articles": danh sách các điều luật liên quan, gồm: 
   		+ "law_id": số hiệu văn bản luật.
   		+ "article_id": điều luật liên quan
   + "question_type": Loại câu hỏi. Gồm: Multiple choice và Yes/No 
   + "question": Câu hỏi dạng văn bản.
   + "choices": danh sách 4 lựa chọn A,B,C,D. Chỉ áp dụng cho câu hỏi dạng multiple choice.
   + "answer": Câu trả lời, gồm các giá trị:
     - Đúng / Sai: Với câu hỏi dạng Yes/No
     - A | B | C | D: với câu hỏi dạng Multiple choice

   Thư mục train_images: chứa các hình ảnh kèm theo 

3. Dữ liệu public test: gồm 2 file json với cấu trúc tương tự như training, nhưng sẽ thiếu đi trường "relevant_articles" và "answer" (với task 1) và trường "answer" (với task 2)
   Thư mục public_test_images: chứa các hình ảnh kèm theo.
