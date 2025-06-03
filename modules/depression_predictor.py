"""Module dự đoán nguy cơ trầm cảm dựa trên phân tích đa chiều."""

import json
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from collections import defaultdict

class DepressionPredictor:
    def __init__(self):
        self.history_file = "data/depression_history.json"
        self.session_file = "data/session_data.json"
        self.assessment_interval = timedelta(hours=24)  # Đánh giá mỗi 24 giờ
        self.last_assessment = None
        self.history = []
        
        # Ngưỡng cảnh báo
        self.thresholds = {
            'negative_emotion_ratio': 0.6,  # 60% cảm xúc tiêu cực
            'emotion_variance': 0.1,  # Hệ số phương sai tối thiểu
            'dislike_ratio': 0.7,  # 70% tỷ lệ dislike
            'usage_duration': 3  # Số giờ sử dụng liên tục
        }
        
        self.load_history()

    def load_history(self):
        """Tải lịch sử đánh giá từ file."""
        try:
            # Tải dữ liệu từ session_data.json nếu có
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                    
                    # Chuyển đổi dữ liệu session thành đánh giá
                    for session in session_data:
                        assessment = {
                            'timestamp': session['timestamp'],
                            'risk_score': float(session.get('risk_score', 0.5)),
                            'risk_level': session.get('risk_level', 'medium'),
                            'metrics': {
                                'emotion_patterns': {
                                    'negative_ratio': float(session['emotions'].get('sad', 0) + 
                                                         session['emotions'].get('angry', 0)),
                                    'emotion_variance': float(np.var(list(session['emotions'].values()))),
                                    'num_records': 1
                                }
                            }
                        }
                        self.history.append(assessment)

            # Tải thêm dữ liệu từ depression_history.json nếu có
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.history.extend(data.get('assessments', []))
                    last_time = data.get('last_assessment')
                    if last_time:
                        self.last_assessment = datetime.fromisoformat(last_time)
                    
            # Sắp xếp theo thời gian và loại bỏ trùng lặp
            if self.history:
                self.history.sort(key=lambda x: x['timestamp'])
                # Giữ bản ghi mới nhất cho mỗi timestamp
                unique_history = {}
                for assessment in self.history:
                    unique_history[assessment['timestamp']] = assessment
                self.history = list(unique_history.values())
                
        except Exception as e:
            logging.error(f"Error loading depression history: {e}")
            self.history = []
            self.last_assessment = None

    def save_history(self):
        """Lưu lịch sử đánh giá."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump({
                    'last_assessment': self.last_assessment.isoformat() if self.last_assessment else None,
                    'assessments': self.history
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving depression history: {e}")

    def analyze_emotion_patterns(self, emotion_history, timeframe_hours=24):
        """Phân tích mô hình cảm xúc trong khoảng thời gian."""
        try:
            if not emotion_history:
                return None
                
            # Lọc dữ liệu trong timeframe
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=timeframe_hours)
            
            recent_emotions = [
                entry for entry in emotion_history 
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]
            
            if not recent_emotions:
                return None
                
            # Tính tỷ lệ cảm xúc tiêu cực
            negative_ratios = []
            variances = []
            
            for entry in recent_emotions:
                emotions = entry['emotions']
                negative_ratio = emotions.get('sad', 0) + emotions.get('angry', 0)
                negative_ratios.append(negative_ratio)
                
                values = list(emotions.values())
                variances.append(np.var(values) if values else 0)
            
            return {
                'negative_ratio': float(np.mean(negative_ratios)),
                'emotion_variance': float(np.mean(variances)),
                'num_records': len(recent_emotions)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing emotion patterns: {e}")
            return None

    def analyze_usage_patterns(self, emotion_history, timeframe_hours=24):
        """Phân tích mẫu hình sử dụng."""
        try:
            if not emotion_history:
                return None
                
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=timeframe_hours)
            
            usage_times = [
                datetime.fromisoformat(entry['timestamp'])
                for entry in emotion_history 
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]
            
            if not usage_times:
                return None
                
            usage_duration = max([
                (t2 - t1).total_seconds() / 3600
                for t1, t2 in zip(usage_times, usage_times[1:])
                if (t2 - t1).total_seconds() <= 3600
            ] + [0])
            
            hour_distribution = defaultdict(int)
            for time in usage_times:
                hour_distribution[time.hour] += 1
                
            late_night_usage = sum(
                count for hour, count in hour_distribution.items()
                if 23 <= hour or hour <= 4
            ) / len(usage_times) if usage_times else 0
            
            return {
                'max_continuous_usage': float(usage_duration),
                'late_night_ratio': float(late_night_usage),
                'num_sessions': len(usage_times)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing usage patterns: {e}")
            return None

    def analyze_interaction_patterns(self, user_preferences):
        """Phân tích mẫu hình tương tác với nội dung."""
        try:
            if not user_preferences:
                return None
                
            total_likes = 0
            total_dislikes = 0
            
            for video_data in user_preferences.values():
                total_likes += video_data.get('likes', 0)
                total_dislikes += video_data.get('dislikes', 0)
                
            total_interactions = total_likes + total_dislikes
            if total_interactions == 0:
                return None
                
            return {
                'dislike_ratio': float(total_dislikes / total_interactions),
                'total_interactions': total_interactions
            }
            
        except Exception as e:
            logging.error(f"Error analyzing interaction patterns: {e}")
            return None

    def calculate_risk_score(self, emotion_data, usage_data, interaction_data):
        """Tính điểm nguy cơ dựa trên các chỉ số."""
        try:
            score = 0
            max_score = 0
            
            if emotion_data:
                # Đánh giá mức độ cảm xúc tiêu cực
                if emotion_data['negative_ratio'] > self.thresholds['negative_emotion_ratio']:
                    score += 3
                elif emotion_data['negative_ratio'] > self.thresholds['negative_emotion_ratio'] * 0.8:
                    score += 2
                max_score += 3
                
                # Đánh giá độ biến thiên cảm xúc
                if emotion_data['emotion_variance'] < self.thresholds['emotion_variance']:
                    score += 2
                max_score += 2
                
            if usage_data:
                # Đánh giá thời gian sử dụng
                if usage_data['max_continuous_usage'] > self.thresholds['usage_duration']:
                    score += 2
                max_score += 2
                
                # Đánh giá sử dụng đêm khuya
                if usage_data['late_night_ratio'] > 0.3:
                    score += 2
                max_score += 2
                
            if interaction_data:
                # Đánh giá tỷ lệ dislike
                if interaction_data['dislike_ratio'] > self.thresholds['dislike_ratio']:
                    score += 2
                elif interaction_data['dislike_ratio'] > self.thresholds['dislike_ratio'] * 0.8:
                    score += 1
                max_score += 2
                
            normalized_score = score / max_score if max_score > 0 else 0
            
            return normalized_score, self._classify_risk(normalized_score)
            
        except Exception as e:
            logging.error(f"Error calculating risk score: {e}")
            return 0, "low"

    def _classify_risk(self, score):
        """Phân loại mức độ nguy cơ."""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        return "low"

    def predict_risk(self, emotion_history, user_preferences=None):
        """Đánh giá nguy cơ trầm cảm."""
        try:
            current_time = datetime.now()
            
            # Kiểm tra xem có cần đánh giá mới không
            if (self.last_assessment and 
                current_time - self.last_assessment < self.assessment_interval):
                # Trả về kết quả đánh giá gần nhất
                if self.history:
                    return (
                        self.history[-1]['risk_score'],
                        self.history[-1]['risk_level']
                    )
                return 0, "low"
            
            # Phân tích các chỉ số
            emotion_patterns = self.analyze_emotion_patterns(emotion_history)
            usage_patterns = self.analyze_usage_patterns(emotion_history)
            interaction_patterns = self.analyze_interaction_patterns(user_preferences)
            
            # Nếu không có đủ dữ liệu cho bất kỳ phân tích nào
            if not any([emotion_patterns, usage_patterns, interaction_patterns]):
                return 0, "low"
            
            # Tính điểm nguy cơ
            risk_score, risk_level = self.calculate_risk_score(
                emotion_patterns, usage_patterns, interaction_patterns
            )
            
            # Lưu kết quả đánh giá
            assessment = {
                'timestamp': current_time.isoformat(),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'metrics': {
                    'emotion_patterns': emotion_patterns,
                    'usage_patterns': usage_patterns,
                    'interaction_patterns': interaction_patterns
                }
            }
            
            self.history.append(assessment)
            self.last_assessment = current_time
            self.save_history()
            
            return risk_score, risk_level
            
        except Exception as e:
            logging.error(f"Error predicting risk: {e}")
            return 0, "low"