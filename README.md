🖐 Hệ thống nhận diện cử tri chỉ trong thời gian thực hiện và chuyển những ký hiệu thành giọng nói 🎤
🚀 Giới thiệu
Dự án này sử dụng MediaPipe, TensorFlow, OpenCV để nhận cử chỉ tay và chuyển đổi chúng thành giọng nói thông qua webcam.
Hệ thống ứng dụng Mạng Neural Nhân tạo (ANN) với kiến ​​trúc Sequential , được huấn luyện để nhận diện 10 ứng cử viên khác nhau , hỗ trợ hỗ trợ giao tiếp tiếp theo cho người Yên lặng hoặc trong các vấn đề đặc biệt.

🎯 Tính năng chính
✅ Nhận được cử chỉ chỉ trong thời gian thực hiện .
✅ Chuyển cử chỉ thành giọng nói để giao tiếp.
✅ Hỗ trợ 10 cuộc bầu cử khác nhau .
✅ Ứng dụng công nghệ AI tiên tiến , giúp nhận diện nhanh và chính xác .

💾 Cài đặt
🛠 Điều kiện tiên quyết
🐍 Python 3.7+ – Trình cài đặt ngôn ngữ chính
💾 RAM 8GB+ – Khuyến nghị để có hiệu suất tối ưu
🖥 CPU 4+ lõi – Để xử lý nhanh hơn
📷 Webcam – Để nhận cử chỉ tay
🎧 Loa – Để phát giọng nói
🎥 Thiết lập dự án
1️⃣ Sao chép dự án
git clone https://github.com/tienbry9999/Nhan-dien-cu-chi-tay-cua-nguoi-khuyet-tat-.git  
cd Hand-Gesture-to-Speech  
2️⃣ Cài đặt các thư viện cần thiết
pip install opencv-python mediapipe numpy tensorflow pandas scikit-learn matplotlib pyautogui pyttsx3  
3️⃣ Thu thập dữ liệu cử chỉ bằng tay
Sử dụng máy tính để quay lại video của từng cử chỉ và lưu vào thư mục cam_xuc.

Trong quá trình này:
✅ Thực hiện 10 cử chỉ tay tương ứng với các cảm xúc:
Bực bội, buồn, đói, ghen tị, thú vị, không thích, lo lắng, tức giận, vui, xấu hổ.
✅ Mỗi video sẽ được gán nhãn cảm xúc tương ứng để phục vụ quá trình huấn luyện mô hình.
✅ Video sẽ được tự động lưu vào thư mục:

D:/AI_IOT/cam_xuc  
✅ Đảm bảo webcam hoạt động tốt , thực hiện xóa chỉ định, đặt đúng vị trí để xác định chính xác hệ thống.

4️⃣ Trích xuất keypoint từ video và lưu vào file CSV
Sau khi thu thập video, hệ thống sẽ sử dụng MediaPipe để xác định mốc trên bàn tay.
Điểm mốc này sẽ được ghi lại vào tệp .csvtrong thư mục extracted_data.

Chạy lệnh sau để thực hiện quá trình này:

python preprocess.py  
Hệ thống sẽ:
✅ Đọc từng video trong thư mục cam_xuc.
✅ Xác định keypoint của bàn tay bằng MediaPipe Holistic .
✅ Lưu mốc thông tin vào các tập tin .csvtrong thư mục extracted_data.

5️⃣ Huấn luyện mô hình nhận dạng cử chỉ
Sau khi có dữ liệu từ tệp CSV, tiến hành huấn luyện mô hình bằng lệnh sau:

python train.py  
Hệ thống sẽ:
✅ Đọc mốc dữ liệu từ các tệp .csv.
✅ Sử dụng Mạng Neural Nhân tạo (ANN) với kiến ​​trúc Sequential để huấn luyện mô hình.
✅ Lưu mô hình huấn luyện dưới dạng hand_emotion_model.keras.

6️⃣ Nhận cử chỉ tay và chuyển đổi thành giọng nói
Sau khi huấn luyện xong, bạn có thể chạy chương trình nhận dạng và chuyển thành giọng nói bằng lệnh sau:

python detect.py  
Hệ thống sẽ:
✅ Sử dụng webcam để quét cử chỉ trong thời gian thực hiện .
✅ Nhận được cảm xúc tương tác dựa trên mô hình đã được huấn luyện.
✅ Chuyển đổi cử chỉ thành giọng nói bằng thư viện pyttsx3.

🛑 Lưu ý
✔ Nhấn 'q' để thoát khỏi chương trình nhận dạng.
✔ Đảm bảo môi trường đủ ánh sáng để nhận dạng chính xác.
✔ Nếu kết quả chưa tốt, bạn có thể thu thập thêm dữ liệu và huấn luyện lại mô hình .

🖐 Hỗ trợ chỉ định
✅ 😠 Bực bội – "Bạn đang cảm thấy dồi dào."
✅ 😞 Buồn – "Bạn đang cảm thấy buồn."
✅ 🍽 Đói – "Bạn đang cảm thấy đói."
✅ 😒 Ghen tị – "Bạn đang cảm thấy ghen tỵ."
✅ 🤩 Hứng thú – "Bạn đang cảm thấy thú vị."
✅ 👎 Không thích – "Bạn đang cảm thấy không thích."
✅ 😟 Lo lắng – "Bạn đang cảm thấy lo lắng."
✅ 😡 Tức giận – "Bạn đang cảm thấy tức giận."
✅ 😀 Vui – "Bạn đang cảm thấy vui."
✅ 😳 Xấu hổ – "Bạn đang cảm thấy xấu hổ."

📝 Giấy phép
© 2025 Nhóm 4 - Lớp CNTT 1603 🎓
🏢 Trường Đại học Đại Nam
