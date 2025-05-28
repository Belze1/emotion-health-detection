"""Emotion Health Detection System modules."""

from .face_detector import FaceDetector
from .emotion_analyzer import EmotionAnalyzer
from .depression_predictor import DepressionPredictor
from .video_recommender import VideoRecommender
from .utils import (
    init_webcam,
    resize_frame,
    draw_text,
    create_emotion_bar,
    save_session_data,
    load_session_data
)

__version__ = '1.0.0'

__all__ = [
    'FaceDetector',
    'EmotionAnalyzer',
    'DepressionPredictor',
    'VideoRecommender',
    'init_webcam',
    'resize_frame',
    'draw_text',
    'create_emotion_bar',
    'save_session_data',
    'load_session_data'
]