"""Module phân tích cảm xúc sử dụng DeepFace."""

from deepface import DeepFace
import numpy as np
import cv2
from typing import Dict, Optional
from datetime import datetime
import json
import os

class EmotionAnalyzer:
    def __init__(self):
        """Khởi tạo emotion analyzer."""
        self.emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral']
        self.emotion_history = []
        self.analysis_interval = 30  # Phân tích mỗi 30 frame
        self.frame_counter = 0
        
    def analyze_emotion(self, face_image: np.ndarray) -> Optional[Dict]:
        """Phân tích cảm xúc từ ảnh khuôn mặt.
        
        Args:
            face_image: Ảnh khuôn mặt đã được cắt
            
        Returns:
            Dict chứa xác suất các cảm xúc hoặc None nếu có lỗi
        """
        try:
            # Tăng bộ đếm frame
            self.frame_counter += 1
            
            # Chỉ phân tích mỗi n frame
            if self.frame_counter % self.analysis_interval != 0:
                return None
                
            # Đảm bảo ảnh ở định dạng BGR
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            
            # Phân tích cảm xúc
            result = DeepFace.analyze(
                face_image,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Trích xuất xác suất cảm xúc
            emotions = {}
            all_emotions = result[0]['emotion']
            total = sum(value for emotion, value in all_emotions.items() 
                       if emotion in self.emotions)
            
            # Chuẩn hóa xác suất cho các cảm xúc được quan tâm
            for emotion in self.emotions:
                if emotion in all_emotions:
                    emotions[emotion] = all_emotions[emotion] / total
                else:
                    emotions[emotion] = 0.0
            
            # Xác định cảm xúc chính
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Lưu kết quả với timestamp
            current_time = datetime.now()
            self.emotion_history.append({
                'timestamp': current_time.isoformat(),
                'emotions': emotions,
                'dominant_emotion': dominant_emotion
            })
            
            # Giữ lại 100 kết quả gần nhất
            if len(self.emotion_history) > 100:
                self.emotion_history.pop(0)
            
            # Lưu lịch sử định kỳ
            if len(self.emotion_history) % 10 == 0:
                self.save_history()
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion
            }
            
        except Exception as e:
            print(f"Lỗi phân tích cảm xúc: {str(e)}")
            return None
    
    def get_emotion_stats(self, minutes: int = 5) -> Dict:
        """Tính toán thống kê cảm xúc trong n phút gần nhất.
        
        Args:
            minutes: Số phút cần phân tích
            
        Returns:
            Dict chứa thống kê cảm xúc
        """
        if not self.emotion_history:
            return {emotion: 0.0 for emotion in self.emotions}
            
        current_time = datetime.now()
        # Lọc dữ liệu trong khoảng thời gian chỉ định
        recent_emotions = [
            entry for entry in self.emotion_history
            if (current_time - datetime.fromisoformat(entry['timestamp'])).total_seconds() <= minutes * 60
        ]
        
        if not recent_emotions:
            return {emotion: 0.0 for emotion in self.emotions}
        
        # Tính trung bình xác suất
        emotion_sums = {emotion: 0.0 for emotion in self.emotions}
        for entry in recent_emotions:
            for emotion, prob in entry['emotions'].items():
                emotion_sums[emotion] += prob
                
        return {
            emotion: value / len(recent_emotions)
            for emotion, value in emotion_sums.items()
        }
    
    def save_history(self, filepath: str = 'data/emotion_history.json'):
        """Lưu lịch sử cảm xúc ra file.
        
        Args:
            filepath: Đường dẫn file lưu lịch sử
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.emotion_history, f)
        except Exception as e:
            print(f"Lỗi lưu lịch sử cảm xúc: {str(e)}")
    
    def load_history(self, filepath: str = 'data/emotion_history.json'):
        """Tải lịch sử cảm xúc từ file.
        
        Args:
            filepath: Đường dẫn file lịch sử
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.emotion_history = json.load(f)
                print(f"Đã tải {len(self.emotion_history)} bản ghi lịch sử cảm xúc")
        except Exception as e:
            print(f"Lỗi tải lịch sử cảm xúc: {str(e)}")
            self.emotion_history = []