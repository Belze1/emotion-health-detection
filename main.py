"""Main application for emotion health detection system."""

import os
import dotenv
dotenv.load_dotenv()

from flask import Flask, render_template, Response, jsonify, request, make_response
from flask_socketio import SocketIO
import cv2
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
import logging
from modules.face_detector import FaceDetector
from modules.emotion_analyzer import EmotionAnalyzer
from modules.depression_predictor import DepressionPredictor
from modules.video_recommender import VideoRecommender
from modules.utils import (
    init_webcam, resize_frame, draw_text,
    save_session_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')
app.config['JSON_SORT_KEYS'] = False


def init_components():
    """Initialize all application components."""
    global face_detector, emotion_analyzer, depression_predictor, video_recommender
    
    try:
        logger.info("Initializing face detector...")
        face_detector = FaceDetector()
        
        logger.info("Initializing emotion analyzer...")
        emotion_analyzer = EmotionAnalyzer()
        
        logger.info("Initializing depression predictor...")
        depression_predictor = DepressionPredictor()
        
        # Initialize video recommender if API key is available
        youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        if youtube_api_key:
            try:
                logger.info("Initializing video recommender...")
                video_recommender = VideoRecommender()
                logger.info("Video recommender initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing video recommender: {str(e)}")
                video_recommender = None
        else:
            logger.warning("No YouTube API key found, video recommendations will be disabled")
            video_recommender = None
            
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        return False

# Global variables
global_cap = None  # Shared webcam instance
cap_lock = threading.Lock()
frame_lock = threading.Lock()
current_emotions = None
last_emotions_display = None  # Store last detected emotions for continuous display
last_risk_check = datetime.now()
last_socket_update = 0
risk_check_interval = int(os.getenv('RISK_CHECK_INTERVAL', 300))
socket_update_interval = float(os.getenv('SOCKET_UPDATE_INTERVAL', 1.0))

# Cấu hình SocketIO
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    logger=False,
    engineio_logger=False
)

def process_frame():
    """Process a single frame from webcam."""
    global current_emotions, last_risk_check, last_socket_update, last_emotions_display
    
    try:
        cap = get_webcam()
        if cap is None:
            return None
            
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
            
        frame = resize_frame(frame)
        frame_with_faces, face_boxes = face_detector.detect_faces(frame)
        
        if face_boxes:
            largest_face = max(face_boxes, key=lambda box: box[2] * box[3])
            face_image = face_detector.extract_face(frame, largest_face)
            
            if face_image is not None:
                result = emotion_analyzer.analyze_emotion(face_image)
                
                if result:
                    current_emotions = result['emotions']
                    
                    # Sắp xếp cảm xúc theo tỷ lệ giảm dần
                    try:
                        sorted_emotions = sorted(
                            current_emotions.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                        
                        # Lấy 2 cảm xúc cao nhất và lưu lại
                        last_emotions_display = sorted_emotions[:2]
                        
                    except Exception as e:
                        logger.error(f"Error processing emotions: {str(e)}")
                    
                    # Throttle socket updates
                    current_time = time.time()
                    if current_time - last_socket_update >= socket_update_interval:
                        try:
                            socketio.emit('emotion_update', {
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'emotions': current_emotions
                            })
                            last_socket_update = current_time
                        except Exception as e:
                            logger.error(f"Error emitting emotion update: {e}")
                    
                    # Check depression risk periodically
                    current_time = datetime.now()
                    if (current_time - last_risk_check).total_seconds() >= risk_check_interval:
                        risk_score, risk_level = depression_predictor.predict_risk(
                            emotion_analyzer.emotion_history,
                            video_recommender.preferences if video_recommender else None
                        )
                        
                        if risk_level != 'low' and video_recommender:
                            recommendations = video_recommender.get_video_recommendations(
                                current_emotions,
                                exclude_videos=[]
                            )
                            try:
                                socketio.emit('recommendations', {
                                    'videos': recommendations,
                                    'risk_score': float(risk_score)
                                })
                            except Exception as e:
                                logger.error(f"Error emitting recommendations: {e}")
                            
                        last_risk_check = current_time
                        save_session_data({
                            'timestamp': current_time.isoformat(),
                            'emotions': current_emotions,
                            'risk_score': float(risk_score),
                            'risk_level': risk_level
                        })
        
        # Always draw emotions if we have them, regardless of current face detection
        if last_emotions_display and len(face_boxes) > 0:
            x, y, w, h = face_boxes[0]  # Use first detected face for text position
            text_x = x + w + 10
            
            # Hiển thị từng cảm xúc và tỷ lệ
            for i, (emotion, ratio) in enumerate(last_emotions_display):
                text_y = y + 30 * (i + 1)
                emotion_text = f"{emotion}: {int(ratio * 100)}%"
                
                if i == 0:  # Cảm xúc cao nhất
                    draw_text(frame_with_faces, emotion_text, (text_x, text_y),
                            font_scale=0.7, thickness=2,
                            text_color=(255, 255, 255), bg_color=(0, 128, 0))
                else:  # Cảm xúc thứ hai
                    draw_text(frame_with_faces, emotion_text, (text_x, text_y),
                            font_scale=0.6, thickness=2,
                            text_color=(255, 255, 255), bg_color=(128, 0, 0))
                    
        return frame_with_faces
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return None

def generate_frames():
    """Generate video frames."""
    while True:
        with frame_lock:
            frame = process_frame()
            
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                )
                
        time.sleep(0.033)  # ~30 FPS

def get_webcam():
    """Get or initialize global webcam instance."""
    global global_cap
    
    with cap_lock:
        if global_cap is None or not global_cap.isOpened():
            logger.info("Initializing webcam...")
            global_cap = init_webcam()
            
            if global_cap is None or not global_cap.isOpened():
                logger.error("Could not connect to webcam")
                return None
                
        return global_cap

@app.route('/')
def index():
    """Render main page."""
    return render_template(
        'index.html',
        video_recommendations_enabled=video_recommender is not None
    )

@app.route('/recommendations')
def recommendations():
    """Show video recommendations page."""
    if not video_recommender:
        return render_template('index.html', error="Video recommendations are not available")
    return render_template(
        'recommendations.html',
        video_recommendations_enabled=True
    )

@app.route('/history')
def history():
    """Show emotion and mental health history page."""
    return render_template('history.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/mental-health-check', methods=['POST'])
def mental_health_check():
    """Thực hiện đánh giá sức khỏe tinh thần."""
    try:
        risk_score, risk_level = depression_predictor.predict_risk(
            emotion_analyzer.emotion_history,
            video_recommender.preferences if video_recommender else None
        )
        
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'risk_score': float(risk_score),
            'risk_level': risk_level
        }
        
        return jsonify(assessment)
    except Exception as e:
        logger.error(f"Error performing mental health check: {str(e)}")
        return make_response(jsonify({'error': 'Internal server error'}), 500)

@app.route('/api/history-data')
def get_history_data():
    """Lấy dữ liệu lịch sử cho biểu đồ."""
    try:
        timeframe = request.args.get('timeframe', '24h')
        
        # Xác định thời điểm bắt đầu dựa trên timeframe
        current_time = datetime.now()
        if timeframe == '24h':
            start_time = current_time - timedelta(hours=24)
        elif timeframe == '7d':
            start_time = current_time - timedelta(days=7)
        elif timeframe == '30d':
            start_time = current_time - timedelta(days=30)
        else:
            return make_response(jsonify({'error': 'Invalid timeframe'}), 400)
        
        # Lấy dữ liệu từ emotion_analyzer và depression_predictor
        emotion_data = [
            entry for entry in emotion_analyzer.emotion_history 
            if datetime.fromisoformat(entry['timestamp']) >= start_time
        ]
        
        assessments = [
            assessment for assessment in depression_predictor.history
            if datetime.fromisoformat(assessment['timestamp']) >= start_time
        ]
        
        # Format dữ liệu cho biểu đồ xu hướng cảm xúc
        emotion_trend_data = {
            'labels': [],
            'negative_ratios': [],
            'variances': []
        }
        
        for entry in emotion_data:
            emotion_trend_data['labels'].append(entry['timestamp'])
            emotions = entry['emotions']
            negative_ratio = emotions.get('sad', 0) + emotions.get('angry', 0)
            emotion_trend_data['negative_ratios'].append(float(negative_ratio))
            
            values = list(emotions.values())
            emotion_trend_data['variances'].append(float(np.var(values)))
        
        # Format dữ liệu cho biểu đồ điểm nguy cơ
        risk_trend_data = {
            'labels': [a['timestamp'] for a in assessments],
            'risk_scores': [float(a['risk_score']) for a in assessments]
        }
        
        # Lấy đánh giá mới nhất
        latest_assessment = assessments[-1] if assessments else None
        
        # Lấy metrics mới nhất
        latest_metrics = {
            'emotion_metrics': depression_predictor.analyze_emotion_patterns(
                emotion_analyzer.emotion_history
            ) or {
                'negative_ratio': 0,
                'emotion_variance': 0,
                'num_records': 0
            },
            'usage_metrics': depression_predictor.analyze_usage_patterns(
                emotion_analyzer.emotion_history
            ) or {
                'max_continuous_usage': 0,
                'late_night_ratio': 0,
                'num_sessions': 0
            },
            'interaction_metrics': depression_predictor.analyze_interaction_patterns(
                video_recommender.preferences if video_recommender else None
            ) or {
                'dislike_ratio': 0,
                'total_interactions': 0
            }
        }
        
        return jsonify({
            'emotion_trend': emotion_trend_data,
            'risk_trend': risk_trend_data,
            'latest_assessment': latest_assessment,
            'latest_metrics': latest_metrics
        })
        
    except Exception as e:
        logger.error(f"Error getting history data: {str(e)}")
        return make_response(jsonify({'error': 'Internal server error'}), 500)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """API endpoint for video recommendations."""
    try:
        if not request.is_json:
            return make_response(jsonify({'error': 'Content-Type must be application/json'}), 400)

        if not video_recommender:
            return make_response(jsonify({'error': 'Video recommendations are not available'}), 503)
            
        data = request.get_json()
        emotion_data = data.get('emotion_data')
        exclude_videos = data.get('exclude_videos', [])

        if not emotion_data:
            return make_response(jsonify({'error': 'Missing emotion_data'}), 400)

        recommendations = video_recommender.get_video_recommendations(
            emotion_data,
            exclude_videos=exclude_videos,
            num_recommendations=8
        )

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return make_response(jsonify({'error': 'Internal server error'}), 500)

@app.route('/api/feedback', methods=['POST'])
def handle_feedback():
    """API endpoint for handling video feedback."""
    try:
        if not request.is_json:
            return make_response(jsonify({'error': 'Content-Type must be application/json'}), 400)

        if not video_recommender:
            return make_response(jsonify({'error': 'Video recommendations are not available'}), 503)
            
        data = request.get_json()
        video_id = data.get('video_id')
        feedback_type = data.get('feedback_type')
        emotion_data = data.get('emotion_data')

        if not all([video_id, feedback_type, emotion_data]):
            return make_response(jsonify({'error': 'Missing required data'}), 400)

        if feedback_type not in ['like', 'dislike']:
            return make_response(jsonify({'error': 'Invalid feedback type'}), 400)

        # Update video preferences
        video_recommender.update_model(video_id, 'like' if feedback_type == 'like' else 'dislike')

        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Error handling feedback: {str(e)}")
        return make_response(jsonify({'error': 'Internal server error'}), 500)

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    if current_emotions:
        try:
            socketio.emit('emotion_update', {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'emotions': current_emotions
            })
        except Exception as e:
            logger.error(f"Error emitting initial emotions: {e}")

def cleanup():
    """Clean up resources."""
    global global_cap
    with cap_lock:
        if global_cap:
            global_cap.release()
            global_cap = None
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        # Create required directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Initialize components
        logger.info("Starting application initialization...")
        if not init_components():
            logger.error("Failed to initialize components")
            exit(1)

        # Load history
        emotion_analyzer.load_history()
        
        # Start server
        logger.info("Starting application at http://localhost:5000")
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False  # Disable reloader to avoid duplicate initialization
        )
    except Exception as e:
        logger.error(f"Application startup error: {str(e)}")
        raise
    finally:
        cleanup()