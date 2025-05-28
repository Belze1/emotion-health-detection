"""Emotion analysis module using DeepFace."""

from deepface import DeepFace
import numpy as np
import cv2
from typing import Dict, Optional
from datetime import datetime
import json
import os

class EmotionAnalyzer:
    def __init__(self):
        """Initialize the emotion analyzer."""
        self.emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral']
        self.emotion_history = []
        self.analysis_interval = 30  # Analyze every 30 frames
        self.frame_counter = 0
        
    def analyze_emotion(self, face_image: np.ndarray) -> Optional[Dict]:
        """Analyze emotions in the face image.
        
        Args:
            face_image: Cropped face image array
            
        Returns:
            Dictionary containing emotion probabilities or None if analysis fails
        """
        try:
            # Increment frame counter
            self.frame_counter += 1
            
            # Only analyze every nth frame
            if self.frame_counter % self.analysis_interval != 0:
                return None
                
            # Ensure image is BGR (DeepFace expects BGR)
            if len(face_image.shape) == 2:  # Convert grayscale to BGR if needed
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            
            # Analyze emotions
            result = DeepFace.analyze(
                face_image,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Extract emotion probabilities
            emotions = {
                emotion: result[0]['emotion'][emotion]
                for emotion in self.emotions
            }
            
            # Normalize probabilities
            total = sum(emotions.values())
            emotions = {k: v/total for k, v in emotions.items()}
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Store result with timestamp
            self.emotion_history.append({
                'timestamp': datetime.now().isoformat(),
                'emotions': emotions,
                'dominant_emotion': dominant_emotion
            })
            
            # Keep only last 100 readings
            if len(self.emotion_history) > 100:
                self.emotion_history.pop(0)
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion
            }
            
        except Exception as e:
            print(f"Error analyzing emotions: {str(e)}")
            return None
    
    def get_emotion_stats(self, minutes: int = 5) -> Dict:
        """Calculate emotion statistics over the last n minutes.
        
        Args:
            minutes: Number of minutes to analyze
            
        Returns:
            Dictionary containing emotion statistics
        """
        if not self.emotion_history:
            return {emotion: 0.0 for emotion in self.emotions}
            
        # Filter readings from last n minutes
        current_time = datetime.now()
        recent_emotions = [
            entry for entry in self.emotion_history
            if (current_time - datetime.fromisoformat(entry['timestamp'])).total_seconds() <= minutes * 60
        ]
        
        if not recent_emotions:
            return {emotion: 0.0 for emotion in self.emotions}
        
        # Calculate average probabilities
        emotion_sums = {emotion: 0.0 for emotion in self.emotions}
        for entry in recent_emotions:
            for emotion, prob in entry['emotions'].items():
                emotion_sums[emotion] += prob
                
        return {
            emotion: value / len(recent_emotions)
            for emotion, value in emotion_sums.items()
        }
    
    def save_history(self, filepath: str = 'data/emotion_history.json'):
        """Save emotion history to file.
        
        Args:
            filepath: Path to save the history file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.emotion_history, f)
    
    def load_history(self, filepath: str = 'data/emotion_history.json'):
        """Load emotion history from file.
        
        Args:
            filepath: Path to the history file
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.emotion_history = json.load(f)