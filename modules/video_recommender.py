"""Video recommendation module using YouTube API and collaborative filtering."""

from googleapiclient.discovery import build
from sklearn.decomposition import TruncatedSVD
import numpy as np
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

class VideoRecommender:
    def __init__(self, api_key: str):
        """Initialize the video recommender.
        
        Args:
            api_key: YouTube Data API key
        """
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.user_preferences = {}
        self.video_cache = {}
        self.emotion_keywords = {
            'happy': ['motivation', 'positive', 'uplifting', 'cheerful', 'funny'],
            'sad': ['calming', 'relaxing', 'meditation', 'peaceful', 'soothing'],
            'angry': ['stress relief', 'calming music', 'nature sounds', 'meditation'],
            'surprise': ['amazing', 'fascinating', 'interesting', 'educational'],
            'neutral': ['entertaining', 'casual', 'lifestyle', 'vlogs']
        }
        self.load_preferences()
        
    def load_preferences(self, filepath: str = 'data/user_preferences.json'):
        """Load user preferences from file.
        
        Args:
            filepath: Path to preferences file
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.user_preferences = data.get('preferences', {})
                    self.video_cache = data.get('video_cache', {})
        except Exception as e:
            print(f"Error loading preferences: {str(e)}")
            
    def save_preferences(self, filepath: str = 'data/user_preferences.json'):
        """Save user preferences to file.
        
        Args:
            filepath: Path to save preferences
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump({
                    'preferences': self.user_preferences,
                    'video_cache': self.video_cache
                }, f)
        except Exception as e:
            print(f"Error saving preferences: {str(e)}")
            
    def search_videos(self, emotion: str, max_results: int = 10) -> List[Dict]:
        """Search for videos based on emotion.
        
        Args:
            emotion: Current dominant emotion
            max_results: Maximum number of videos to return
            
        Returns:
            List of video information dictionaries
        """
        try:
            # Get keywords for emotion
            keywords = self.emotion_keywords.get(emotion, ['entertaining'])
            
            # Randomly select keywords
            search_query = np.random.choice(keywords)
            
            # Search for videos
            request = self.youtube.search().list(
                part='snippet',
                q=search_query,
                type='video',
                videoDuration='short',  # Only short videos
                maxResults=max_results
            )
            response = request.execute()
            
            videos = []
            for item in response['items']:
                video_id = item['id']['videoId']
                video_info = {
                    'id': video_id,
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                    'emotion_context': emotion
                }
                
                # Cache video info
                self.video_cache[video_id] = video_info
                videos.append(video_info)
                
            return videos
            
        except Exception as e:
            print(f"Error searching videos: {str(e)}")
            return []
            
    def update_preferences(self, video_id: str, liked: bool):
        """Update user preferences for a video.
        
        Args:
            video_id: YouTube video ID
            liked: Whether user liked the video
        """
        if video_id not in self.user_preferences:
            self.user_preferences[video_id] = {
                'rating': 1 if liked else -1,
                'timestamp': datetime.now().isoformat(),
                'emotion_context': self.video_cache.get(video_id, {}).get('emotion_context')
            }
        else:
            self.user_preferences[video_id]['rating'] = 1 if liked else -1
            self.user_preferences[video_id]['timestamp'] = datetime.now().isoformat()
            
        self.save_preferences()
        
    def get_collaborative_recommendations(self, emotion: str, n_recommendations: int = 5) -> List[Dict]:
        """Get video recommendations using collaborative filtering.
        
        Args:
            emotion: Current emotion for context
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended video information dictionaries
        """
        try:
            if not self.user_preferences:
                return self.search_videos(emotion, n_recommendations)
                
            # Create user-video matrix
            video_ids = list(self.video_cache.keys())
            ratings = []
            
            for video_id in video_ids:
                if video_id in self.user_preferences:
                    ratings.append(self.user_preferences[video_id]['rating'])
                else:
                    ratings.append(0)
                    
            # Perform SVD
            svd = TruncatedSVD(n_components=min(len(video_ids), 10))
            video_features = svd.fit_transform(np.array(ratings).reshape(1, -1))
            
            # Get similar videos
            similarities = []
            for i, video_id in enumerate(video_ids):
                # Consider emotion context
                emotion_match = (
                    self.video_cache[video_id].get('emotion_context') == emotion
                )
                similarity = np.dot(video_features[0], svd.components_[:, i])
                if emotion_match:
                    similarity *= 1.5  # Boost emotion-matching videos
                    
                similarities.append((video_id, similarity))
                
            # Sort by similarity
            recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)
            
            # Get video information
            recommended_videos = []
            seen_videos = set(self.user_preferences.keys())
            
            for video_id, _ in recommendations:
                if video_id not in seen_videos and video_id in self.video_cache:
                    recommended_videos.append(self.video_cache[video_id])
                    if len(recommended_videos) >= n_recommendations:
                        break
                        
            # If not enough recommendations, supplement with search
            if len(recommended_videos) < n_recommendations:
                additional = self.search_videos(
                    emotion,
                    n_recommendations - len(recommended_videos)
                )
                recommended_videos.extend(additional)
                
            return recommended_videos
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return self.search_videos(emotion, n_recommendations)  # Fallback to search