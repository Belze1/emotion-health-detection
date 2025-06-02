# Emotion Health Detection

Hệ thống phân tích cảm xúc và đề xuất video dựa trên trạng thái cảm xúc.

## Tính Năng Mới

- Cải thiện hệ thống gợi ý video:
  - Phân tích cảm xúc thời gian thực
  - Tìm kiếm video phù hợp dựa trên trạng thái cảm xúc
  - Sử dụng embeddings để so sánh ngữ nghĩa
  - Cập nhật gợi ý real-time qua WebSocket

## Cài Đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Cấu hình môi trường:
```bash
cp .env.example .env
```
- Thêm YouTube API key vào file .env
- Điều chỉnh các thông số khác nếu cần

3. Chuẩn bị dữ liệu:
```bash
mkdir -p data models/embeddings
# Copy file CSV vào thư mục data/
```

4. Chạy ứng dụng:
```bash
python main.py
```

## Cấu Trúc Thư Mục

```
.
├── data/                      # Dữ liệu
│   └── cleaned_youtube_metadata.csv
├── models/                    # Model và cache
│   └── embeddings/
├── modules/                   # Code chính
│   ├── emotion_analyzer.py
│   ├── face_detector.py
│   ├── video_recommender.py
│   └── ...
├── static/                    # Assets
│   ├── css/
│   └── js/
├── templates/                 # HTML templates
├── .env                      # Cấu hình
├── .env.example              # Template cấu hình
└── main.py                   # Entry point
```

## Yêu Cầu Hệ Thống

- Python 3.8+
- Webcam
- YouTube API key
- Thư viện: xem requirements.txt

## Sử Dụng

1. Mở http://localhost:5000
2. Cho phép truy cập webcam
3. Hệ thống sẽ:
   - Phân tích cảm xúc qua webcam
   - Hiển thị biểu đồ cảm xúc realtime
   - Gợi ý video phù hợp với trạng thái

## Lưu Ý

- Đảm bảo file CSV chứa dữ liệu video đúng định dạng
- YouTube API key cần có quyền truy cập API Data
- Cache embeddings được lưu tại models/embeddings/