"""Depression risk prediction module using emotion patterns."""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class DepressionPredictor:
    def __init__(self, model_path: str = 'models/depression_model.pkl'):
        """Initialize the depression predictor.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.risk_threshold = 0.7  # High risk threshold
        self.load_model()
        
    def load_model(self):
        """Load the trained model if it exists, otherwise create a new one."""
        try:
            if os.path.exists(self.model_path):
                loaded = joblib.load(self.model_path)
                self.model = loaded['model']
                self.scaler = loaded['scaler']
            else:
                # Initialize new model if no saved model exists
                self.model = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    solver='adam',
                    max_iter=1000
                )
                print("Created new model - training required")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            
    def save_model(self):
        """Save the current model and scaler."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, self.model_path)
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            
    def extract_features(self, emotion_history: List[Dict]) -> Optional[np.ndarray]:
        """Extract features from emotion history for prediction.
        
        Args:
            emotion_history: List of emotion readings with timestamps
            
        Returns:
            Feature vector as numpy array or None if extraction fails
        """
        try:
            if not emotion_history:
                return None
                
            # Get emotions from most recent entry
            emotions = list(emotion_history[-1]['emotions'].keys())
            
            # Calculate features
            features = []
            
            # Average emotion probabilities
            emotion_sums = {e: 0.0 for e in emotions}
            for entry in emotion_history:
                for emotion, prob in entry['emotions'].items():
                    emotion_sums[emotion] += prob
            avg_emotions = [value / len(emotion_history) for value in emotion_sums.values()]
            features.extend(avg_emotions)
            
            # Emotional volatility (std dev of probabilities)
            volatility = []
            for emotion in emotions:
                probs = [entry['emotions'][emotion] for entry in emotion_history]
                volatility.append(np.std(probs))
            features.extend(volatility)
            
            # Negative emotion ratio
            negative_emotions = ['sad', 'angry']
            total_negative = sum(emotion_sums[e] for e in negative_emotions)
            total_all = sum(emotion_sums.values())
            negative_ratio = total_negative / total_all if total_all > 0 else 0
            features.append(negative_ratio)
            
            # Emotion transitions
            transitions = 0
            for i in range(1, len(emotion_history)):
                if emotion_history[i]['dominant_emotion'] != emotion_history[i-1]['dominant_emotion']:
                    transitions += 1
            transition_rate = transitions / len(emotion_history)
            features.append(transition_rate)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
            
    def predict_risk(self, emotion_history: List[Dict]) -> Tuple[float, bool]:
        """Predict depression risk from emotion history.
        
        Args:
            emotion_history: List of emotion readings with timestamps
            
        Returns:
            Tuple of (risk_score, is_high_risk)
        """
        try:
            # Extract features
            features = self.extract_features(emotion_history)
            if features is None:
                return 0.0, False
                
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probability
            risk_score = self.model.predict_proba(features_scaled)[0][1]
            is_high_risk = risk_score >= self.risk_threshold
            
            return risk_score, is_high_risk
            
        except Exception as e:
            print(f"Error predicting risk: {str(e)}")
            return 0.0, False
            
    def train(self, emotion_histories: List[List[Dict]], labels: List[int]):
        """Train the model on labeled emotion histories.
        
        Args:
            emotion_histories: List of emotion history sequences
            labels: Binary labels (0: no depression, 1: depression)
        """
        try:
            # Extract features from all histories
            features = []
            for history in emotion_histories:
                feature_vector = self.extract_features(history)
                if feature_vector is not None:
                    features.append(feature_vector.flatten())
                    
            if not features:
                print("No valid features extracted for training")
                return
                
            features = np.array(features)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled, labels)
            
            # Save trained model
            self.save_model()
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            
    def update_model(self, emotion_history: List[Dict], label: int):
        """Update model with new training example.
        
        Args:
            emotion_history: New emotion history sequence
            label: Binary label (0: no depression, 1: depression)
        """
        try:
            # Extract features
            features = self.extract_features(emotion_history)
            if features is None:
                return
                
            # Update scaler and transform features
            features_scaled = self.scaler.partial_fit(features).transform(features)
            
            # Update model
            self.model.partial_fit(features_scaled, [label], classes=[0, 1])
            
            # Save updated model
            self.save_model()
            
        except Exception as e:
            print(f"Error updating model: {str(e)}")