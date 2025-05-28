# Emotion Health Detection System

Hệ thống phát hiện và phân tích cảm xúc thời gian thực từ webcam, với các tính năng:
- Nhận diện khuôn mặt
- Phân tích cảm xúc (hạnh phúc, buồn, giận dữ, ngạc nhiên, trung tính)
- Theo dõi cảm xúc thông qua biểu đồ thời gian thực
- Phát hiện nguy cơ trầm cảm
- Đề xuất video dựa trên trạng thái cảm xúc

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/Belze1/emotion-health-detection.git
cd emotion-health-detection
```

2. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# HOẶC
venv\Scripts\activate  # Windows
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

4. Tạo file .env từ mẫu .env.example:
```bash
cp .env.example .env
```

5. Cập nhật các biến môi trường trong file .env:
- YOUTUBE_API_KEY: API key của YouTube (để sử dụng tính năng đề xuất video)
- Các cài đặt khác như độ phân giải webcam, khoảng thời gian kiểm tra, v.v.

## Sử dụng

1. Khởi động ứng dụng:
```bash
python main.py
```

2. Truy cập ứng dụng tại: http://localhost:5000

## Tính năng

### 1. Nhận diện khuôn mặt và cảm xúc
- Sử dụng MediaPipe để phát hiện khuôn mặt
- Phân tích cảm xúc thời gian thực
- Hiển thị kết quả trực tiếp trên video stream

### 2. Phân tích cảm xúc
- Biểu đồ thời gian thực hiển thị các cảm xúc
- Theo dõi xu hướng cảm xúc theo thời gian
- Stacked area chart cho phép quan sát tỷ lệ các cảm xúc

### 3. Phát hiện trầm cảm
- Phân tích mẫu cảm xúc để phát hiện nguy cơ
- Cảnh báo khi phát hiện dấu hiệu rủi ro
- Lưu trữ lịch sử để theo dõi dài hạn

### 4. Đề xuất video
- Dựa trên trạng thái cảm xúc hiện tại
- Tích hợp với YouTube API
- Hệ thống đánh giá và phản hồi

## Cấu trúc project

```
emotion-health-detection/
├── modules/
│   ├── face_detector.py        # Nhận diện khuôn mặt
│   ├── emotion_analyzer.py     # Phân tích cảm xúc
│   ├── depression_predictor.py # Dự đoán trầm cảm
│   ├── video_recommender.py    # Đề xuất video
│   └── utils.py               # Công cụ hỗ trợ
├── static/
│   ├── css/                   # Styles
│   └── js/                    # JavaScript
├── templates/                 # HTML templates
├── main.py                   # Ứng dụng chính
├── requirements.txt          # Dependencies
└── .env                     # Cấu hình
```

## Công nghệ sử dụng

- **Backend**: Python, Flask, SocketIO
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: TensorFlow, DeepFace
- **APIs**: YouTube Data API

## Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo issues hoặc pull requests để cải thiện project.