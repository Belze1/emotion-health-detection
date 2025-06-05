"""Reinforcement Learning Agent for video recommendation."""

import numpy as np
from typing import Dict, List, Tuple
import json
import os
from datetime import datetime
import logging

class RL_Agent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """Initialize RL agent with hyperparameters."""
        self.lr = learning_rate
        self.gamma = discount_factor  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.q_values = {}  # video_id -> q_value
        
    def get_state_vector(self, emotions: Dict[str, float]) -> np.ndarray:
        """Convert emotions dictionary to state vector."""
        return np.array([
            emotions.get('happy', 0),
            emotions.get('sad', 0),
            emotions.get('angry', 0),
            emotions.get('surprise', 0),
            emotions.get('neutral', 0)
        ])
        
    def calculate_reward(self, feedback_type: str, 
                        watch_duration: float,
                        emotion_before: Dict[str, float],
                        emotion_after: Dict[str, float]) -> float:
        """Calculate reward based on user interaction and emotion change."""
        # Base reward from feedback
        reward = 1.0 if feedback_type == 'like' else -1.0
        
        # Bonus for watch duration (normalized by 5 minutes)
        duration_reward = min(watch_duration / 300.0, 1.0)
        
        # Emotion change reward
        # Positive reward for increase in happy/neutral, decrease in sad/angry
        emotion_weights = {
            'happy': 1.0,
            'neutral': 0.5,
            'sad': -1.0,
            'angry': -1.0,
            'surprise': 0.0
        }
        
        emotion_change = sum(
            emotion_weights[emotion] * (emotion_after[emotion] - emotion_before[emotion])
            for emotion in emotion_weights
        )
        
        # Combine rewards with weights
        total_reward = (
            0.5 * reward +  # Feedback weight
            0.2 * duration_reward +  # Duration weight
            0.3 * emotion_change  # Emotion change weight
        )
        
        return total_reward
        
    def update(self, video_id: str, 
              state: Dict[str, float],
              next_state: Dict[str, float],
              reward: float):
        """Update Q-value for video using Q-learning."""
        if video_id not in self.q_values:
            self.q_values[video_id] = 0.0
            
        state_vector = self.get_state_vector(state)
        next_state_vector = self.get_state_vector(next_state)
        
        # Get maximum Q-value for next state
        max_next_q = max(self.q_values.values()) if self.q_values else 0
        
        # Q-learning update rule
        current_q = self.q_values[video_id]
        self.q_values[video_id] = current_q + self.lr * (
            reward + 
            self.gamma * max_next_q - 
            current_q
        )
        
    def select_action(self, candidates: List[str], state: Dict[str, float]) -> List[str]:
        """Select videos using epsilon-greedy policy."""
        if len(candidates) == 0:
            return []
            
        if np.random.random() < self.epsilon:
            # Exploration: random selection
            selected = list(np.random.choice(
                candidates,
                size=min(4, len(candidates)),
                replace=False
            ))
        else:
            # Exploitation: select highest Q-value videos
            sorted_videos = sorted(
                candidates,
                key=lambda vid: self.q_values.get(vid, 0.0),
                reverse=True
            )
            selected = sorted_videos[:4]
            
        return selected
        
    def save_state(self, video_id: str,
                   state: Dict[str, float],
                   watch_duration: float,
                   reward: float,
                   filepath='data/user_preferences.json'):
        """Save state history to preferences file."""
        try:
            # Load existing preferences
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    preferences = json.load(f)
            else:
                preferences = {}
                
            # Update video entry
            if video_id not in preferences:
                preferences[video_id] = {
                    'likes': 0,
                    'dislikes': 0,
                    'last_updated': None,
                    'state_history': [],
                    'q_value': self.q_values.get(video_id, 0.0)
                }
                
            # Add new state to history
            preferences[video_id]['state_history'].append({
                'timestamp': datetime.now().isoformat(),
                'emotion_state': state,
                'watch_duration': watch_duration,
                'reward': reward
            })
            
            # Update Q-value
            preferences[video_id]['q_value'] = self.q_values.get(video_id, 0.0)
            
            # Save updated preferences
            with open(filepath, 'w') as f:
                json.dump(preferences, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving RL state: {str(e)}")
            
    def load_state(self, filepath='data/user_preferences.json'):
        """Load Q-values from preferences file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    preferences = json.load(f)
                    
                # Load Q-values
                for video_id, data in preferences.items():
                    self.q_values[video_id] = data.get('q_value', 0.0)
                    
        except Exception as e:
            logging.error(f"Error loading RL state: {str(e)}")