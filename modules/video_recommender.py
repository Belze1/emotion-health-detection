"""Video recommendation module using embeddings and reinforcement learning."""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pickle
import json
import time
import logging
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
load_dotenv()

class VideoRecommender:
    def __init__(self, csv_path='data/cleaned_youtube_metadata.csv', cache_dir='models/embeddings'):
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.emotion_history = []
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            logging.warning("YouTube API key not found")
        else:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        
        logging.info(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(self.embedding_model_name)
        
        self.load_data()
        self.create_or_load_embeddings()
        self.preferences = self.load_preferences()
            
    def load_data(self):
        try:
            if not os.path.exists(self.csv_path):
                logging.error(f"CSV file not found: {self.csv_path}")
                self.df = pd.DataFrame()
                return

            self.df = pd.read_csv(self.csv_path)
            # Remove rows with empty or invalid content
            self.df = self.df[self.df['cleaned_content'].notna() & (self.df['cleaned_content'].str.len() > 0)]
            self.df = self.df.reset_index(drop=True)
            logging.info(f"Loaded {len(self.df)} valid videos from CSV")
            
        except Exception as e:
            logging.error(f"Error loading CSV: {str(e)}")
            self.df = pd.DataFrame()
            
    def create_or_load_embeddings(self):
        cache_path = os.path.join(self.cache_dir, f"video_embeddings_{self.embedding_model_name}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.video_embeddings = cached_data if isinstance(cached_data, dict) else {}
                logging.info(f"Loaded {len(self.video_embeddings)} embeddings from cache")
            except:
                self._generate_embeddings(cache_path)
        else:
            self._generate_embeddings(cache_path)
            
    def _generate_embeddings(self, cache_path):
        self.video_embeddings = {}
        batch_size = 32
        
        total_videos = len(self.df)
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
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.video_embeddings, f)
            logging.info("Saved embeddings to cache")
        except Exception as e:
            logging.error(f"Error saving cache: {str(e)}")
        
    def encode_emotions(self, emotions):
        try:
            # Get dominant emotion
            dominant = max(emotions.items(), key=lambda x: x[1])[0]
            strength = emotions[dominant]
            
            # Create search query based on emotions
            if dominant == 'happy' and strength > 0.5:
                return "I want to watch something fun happy entertaining positive uplifting cheerful video"
            elif dominant == 'sad' and strength > 0.5:
                return "I want to watch something motivational inspiring encouraging uplifting hopeful video"
            elif dominant == 'angry' and strength > 0.5:
                return "I want to watch something calming peaceful relaxing meditative soothing video"
            elif dominant == 'surprise' and strength > 0.5:
                return "I want to watch something amazing fascinating incredible interesting exciting video"
            else:
                return "I want to watch something entertaining educational interesting engaging informative video"
                
        except Exception as e:
            logging.error(f"Error encoding emotions: {str(e)}")
            return "I want to watch something entertaining and interesting"

    def get_video_recommendations(self, emotions, num_candidates=20, num_recommendations=4):
        try:
            if len(self.video_embeddings) == 0:
                logging.error("No video embeddings available")
                return []
                
            # Get emotion query
            query = self.encode_emotions(emotions)
            logging.info(f"Emotion query: {query}")
            
            # Get query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarities
            similarities = {}
            for video_id, video_embedding in self.video_embeddings.items():
                try:
                    sim = 1 - cosine(query_embedding, video_embedding)
                    similarities[video_id] = sim
                except:
                    continue
                    
            # Get top videos
            if not similarities:
                logging.warning("No similarities calculated")
                return []
                
            top_videos = []
            for video_id, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]:
                try:
                    video_data = self.df[self.df['video_id'] == video_id].iloc[0]
                    video_info = video_data.to_dict()
                    video_info['similarity'] = float(sim)
                    
                    if hasattr(self, 'youtube'):
                        video_info['thumbnail'] = f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"
                        
                    top_videos.append(video_info)
                    logging.info(f"Selected video: {video_id} (similarity: {sim:.3f})")
                except Exception as e:
                    logging.error(f"Error processing video {video_id}: {str(e)}")
                    continue
                    
            logging.info(f"Found {len(top_videos)} recommendations")
            return top_videos

        except Exception as e:
            logging.error(f"Error getting recommendations: {str(e)}")
            return []
            
    def update_model(self, video_id, feedback_type):
        """Cập nhật preference theo phản hồi"""
        try:
            if feedback_type == 'like':
                self.update_preference(video_id, liked=True)
            elif feedback_type == 'dislike':
                self.update_preference(video_id, liked=False)
        except Exception as e:
            logging.error(f"Error updating model: {str(e)}")
            
    def update_preference(self, video_id, liked=True):
        if video_id not in self.preferences:
            self.preferences[video_id] = {
                'likes': 0,
                'dislikes': 0,
                'last_updated': None
            }
            
        pref = self.preferences[video_id]
        if liked:
            pref['likes'] += 1
        else:
            pref['dislikes'] += 1
            
        pref['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.save_preferences()
            
    def load_preferences(self, filepath='data/user_preferences.json'):
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Error loading preferences: {str(e)}")
            return {}
            
    def save_preferences(self, filepath='data/user_preferences.json'):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.preferences, f)
        except Exception as e:
            logging.error(f"Error saving preferences: {str(e)}")
