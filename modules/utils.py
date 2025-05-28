"""Utility functions for the emotion health detection system."""

import cv2
import numpy as np
from typing import Tuple, Optional
import json
import os
from datetime import datetime

def init_webcam(width: int = 640, height: int = 480) -> Optional[cv2.VideoCapture]:
    """Initialize webcam with specified resolution."""
    try:
        for index in range(2):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Đã kết nối webcam tại index {index}")
                    return cap
            
        raise Exception("Không tìm thấy webcam")
        
    except Exception as e:
        print(f"Lỗi khi khởi tạo webcam: {str(e)}")
        if 'cap' in locals():
            cap.release()
        return None

def resize_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    """Resize frame while maintaining aspect ratio."""
    height = int(frame.shape[0] * (width / frame.shape[1]))
    return cv2.resize(frame, (width, height))

def draw_text(frame: np.ndarray, text: str, position: Tuple[int, int],
              font_scale: float = 0.6, thickness: int = 2,
              text_color: Tuple[int, int, int] = (255, 255, 255),
              bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Draw text with background on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline),
        (x + text_width, y + baseline),
        bg_color,
        -1
    )
    
    cv2.putText(
        frame, text, position, font,
        font_scale, text_color, thickness
    )
    
    return frame

def create_emotion_bar(emotions: dict, width: int = 300, height: int = 400) -> np.ndarray:
    """Create bar chart visualization of emotions."""
    # Tạo ảnh nền đen
    chart = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sắp xếp cảm xúc theo thứ tự giảm dần
    sorted_emotions = sorted(
        emotions.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Lấy 3 cảm xúc mạnh nhất
    top_emotions = sorted_emotions[:3]
    
    # Màu sắc cho từng cảm xúc
    emotion_colors = {
        'happy': (0, 255, 0),     # Xanh lá
        'sad': (255, 0, 0),       # Đỏ
        'angry': (0, 0, 255),     # Xanh dương
        'surprise': (255, 255, 0), # Vàng
        'neutral': (128, 128, 128) # Xám
    }
    
    # Tạo thanh hiển thị cho mỗi cảm xúc
    bar_height = height // 4  # Chiều cao mỗi thanh
    spacing = height // 8     # Khoảng cách giữa các thanh
    
    y_position = spacing
    for emotion, prob in top_emotions:
        # Tính chiều dài thanh
        bar_length = int(prob * (width - 100))
        
        # Vẽ thanh
        color = emotion_colors.get(emotion, (255, 255, 255))
        cv2.rectangle(chart, 
                     (0, y_position), 
                     (bar_length, y_position + bar_height),
                     color, -1)
        
        # Viết tên cảm xúc và phần trăm
        text = f"{emotion}: {int(prob * 100)}%"
        draw_text(
            chart, text,
            (bar_length + 10, y_position + bar_height // 2),
            font_scale=0.7,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0)
        )
        
        y_position += bar_height + spacing
    
    return chart

def save_session_data(data: dict, filepath: str = 'data/session_data.json'):
    """Save session data to file."""
    try:
        data['timestamp'] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(data)
            else:
                existing_data = [existing_data, data]
        else:
            existing_data = [data]
            
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu: {str(e)}")

def load_session_data(filepath: str = 'data/session_data.json') -> list:
    """Load session data from file."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu: {str(e)}")
        return []