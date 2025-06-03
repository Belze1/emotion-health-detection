"""Video recommendation module based on emotional state."""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pickle
import json
import logging
from datetime import datetime
from googleapiclient.discovery import build

logging.basicConfig(level=logging.INFO)

class VideoRecommender:
    def __init__(self, csv_path='data/cleaned_youtube_metadata.csv', cache_dir='models'):
        logging.info("Initializing VideoRecommender...")
        
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.video_embeddings = {}
        
        # Tạo thư mục cache nếu chưa tồn tại
        os.makedirs(self.cache_dir, exist_ok=True)

        # Tải preferences trước khi tải model
        self.preferences = self.load_preferences()
        
        # Khởi tạo YouTube API client
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            logging.warning("YouTube API key not found")
            self.youtube = None
        else:
            try:
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
                logging.info("YouTube API client initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing YouTube API client: {str(e)}")
                self.youtube = None

        try:
            # Tải model embedding
            logging.info(f"Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            logging.info("Embedding model loaded successfully")

            # Tải dữ liệu và tạo embeddings
            self.load_data()
            self.create_or_load_embeddings()
            
            logging.info("VideoRecommender initialization completed")
            
        except Exception as e:
            logging.error(f"Error during VideoRecommender initialization: {str(e)}")
            raise
            
    def load_data(self):
        """Tải dữ liệu video từ CSV."""
        try:
            if not os.path.exists(self.csv_path):
                logging.error(f"CSV file not found: {self.csv_path}")
                self.df = pd.DataFrame()
                return

            logging.info(f"Loading data from {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
            self.df = self.df[self.df['cleaned_content'].notna() & (self.df['cleaned_content'].str.len() > 0)]
            self.df = self.df.reset_index(drop=True)
            logging.info(f"Loaded {len(self.df)} valid videos from CSV")
            
        except Exception as e:
            logging.error(f"Error loading CSV: {str(e)}")
            self.df = pd.DataFrame()
            
    def create_or_load_embeddings(self):
        """Tạo hoặc tải embeddings từ cache."""
        cache_path = os.path.join(self.cache_dir, f"video_embeddings_{self.embedding_model_name}.pkl")
        
        if os.path.exists(cache_path):
            try:
                logging.info("Loading embeddings from cache...")
                with open(cache_path, 'rb') as f:
                    self.video_embeddings = pickle.load(f)
                logging.info(f"Loaded {len(self.video_embeddings)} embeddings from cache")
                return
            except Exception as e:
                logging.error(f"Error loading embeddings cache: {str(e)}")
        
        self._generate_embeddings(cache_path)
            
    def _generate_embeddings(self, cache_path):
        """Tạo embeddings cho nội dung video."""
        try:
            logging.info("Generating new embeddings...")
            batch_size = 32
            
            total_videos = len(self.df)
            if total_videos == 0:
                logging.warning("No videos to generate embeddings for")
                return
                
            logging.info(f"Generating embeddings for {total_videos} videos")
            
            for i in range(0, total_videos, batch_size):
                batch_df = self.df.iloc[i:i+batch_size]
                try:
                    embeddings = self.model.encode(batch_df['cleaned_content'].tolist())
                    for j, row in batch_df.iterrows():
                        self.video_embeddings[row['video_id']] = embeddings[j-i]
                except Exception as e:
                    logging.error(f"Error generating embeddings for batch {i}: {str(e)}")
                    continue
                    
                if i % 100 == 0:
                    logging.info(f"Progress: {i}/{total_videos} videos processed")
                    
            logging.info(f"Generated embeddings for {len(self.video_embeddings)} videos")
            
            # Save embeddings to cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.video_embeddings, f)
                logging.info("Saved embeddings to cache")
            except Exception as e:
                logging.error(f"Error saving embeddings cache: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error in _generate_embeddings: {str(e)}")

    def encode_emotions(self, emotions):
        """Chuyển đổi trạng thái cảm xúc thành text description."""
        try:
            # Kiểm tra emotions hợp lệ
            if not emotions or not isinstance(emotions, dict):
                raise ValueError("Invalid emotions data")

            # Sắp xếp cảm xúc theo tỷ lệ giảm dần
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            
            description = ["Content that combines"]
            emotion_map = {
                'happy': 'positive and uplifting',
                'sad': 'motivational and inspiring',
                'angry': 'calming and peaceful',
                'surprise': 'amazing and fascinating',
                'neutral': 'balanced and informative'
            }
            
            significant_emotions = [
                (emotion, ratio) for emotion, ratio in sorted_emotions 
                if ratio > 0.1  # Chỉ xét cảm xúc > 10%
            ]
            
            if not significant_emotions:
                description.append("engaging and entertaining content")
            else:
                for emotion, ratio in significant_emotions:
                    description.append(f"{emotion_map.get(emotion, 'engaging')} content ({int(ratio*100)}%)")
            
            final_text = " with ".join(description)
            logging.info(f"Generated description: {final_text}")
            
            # Encode và normalize
            embedding = self.model.encode([final_text])[0]
            return embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            logging.error(f"Error encoding emotions: {str(e)}")
            # Fallback to default embedding
            default_text = "engaging and entertaining content"
            embedding = self.model.encode([default_text])[0]
            return embedding / np.linalg.norm(embedding)

    def get_video_recommendations(self, emotions, exclude_videos=None, num_recommendations=4):
        """Get video recommendations based on emotional state."""
        try:
            if len(self.video_embeddings) == 0:
                logging.error("No video embeddings available")
                return []

            if not emotions:
                logging.error("No emotion data provided")
                return []

            exclude_set = set(exclude_videos) if exclude_videos else set()
            
            # Encode emotions
            query_embedding = self.encode_emotions(emotions)
            
            # Calculate similarities
            similarities = {}
            for video_id, video_embedding in self.video_embeddings.items():
                if video_id in exclude_set:
                    continue
                    
                try:
                    sim = float(1 - cosine(query_embedding, video_embedding))
                    if not np.isnan(sim):
                        similarities[video_id] = sim
                except:
                    continue
                    
            if not similarities:
                logging.warning("No similarities calculated")
                return []
            
            # Apply preference adjustments
            adjusted_similarities = {}
            for video_id, sim in similarities.items():
                pref = self.preferences.get(video_id, {})
                likes = pref.get('likes', 0)
                dislikes = pref.get('dislikes', 0)
                
                # Điều chỉnh điểm dựa trên lịch sử tương tác
                if likes + dislikes > 0:
                    like_ratio = likes / (likes + dislikes)
                    sim = sim * (0.8 + 0.4 * like_ratio)  # Điều chỉnh ±20% dựa trên tỷ lệ like
                
                adjusted_similarities[video_id] = sim
            
            # Get top recommendations
            recommendations = []
            for video_id, sim in sorted(adjusted_similarities.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]:
                try:
                    video_data = self.df[self.df['video_id'] == video_id].iloc[0]
                    recommendations.append({
                        'video_id': str(video_data['video_id']),
                        'title': str(video_data['title']),
                        'similarity': float(sim)
                    })
                except Exception as e:
                    logging.error(f"Error processing video {video_id}: {str(e)}")
                    continue
                    
            logging.info(f"Found {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}")
            return []
            
    def update_model(self, video_id, feedback_type):
        """Update user preferences based on feedback."""
        try:
            if video_id not in self.preferences:
                self.preferences[video_id] = {
                    'likes': 0,
                    'dislikes': 0,
                    'last_updated': None
                }
            
            pref = self.preferences[video_id]
            if feedback_type == 'like':
                pref['likes'] += 1
            else:
                pref['dislikes'] += 1
                
            pref['last_updated'] = datetime.now().isoformat()
            self.save_preferences()
            
        except Exception as e:
            logging.error(f"Error updating model: {str(e)}")

    def load_preferences(self, filepath='data/user_preferences.json'):
        """Load user preferences from file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading preferences: {str(e)}")
            return {}
            
    def save_preferences(self, filepath='data/user_preferences.json'):
        """Save user preferences to file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving preferences: {str(e)}")
