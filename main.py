"""Main application for emotion health detection system."""

import os
import dotenv
dotenv.load_dotenv()

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from datetime import datetime
import json
import threading
import time
from modules.face_detector import FaceDetector
from modules.emotion_analyzer import EmotionAnalyzer
from modules.depression_predictor import DepressionPredictor
from modules.video_recommender import VideoRecommender
from modules.utils import (
    init_webcam, resize_frame, draw_text,
    save_session_data
)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
global_cap = None  # Shared webcam instance
cap_lock = threading.Lock()
frame_lock = threading.Lock()
current_emotions = None
last_risk_check = datetime.now()
risk_check_interval = int(os.getenv('RISK_CHECK_INTERVAL', 300))
last_socket_update = 0  # Throttle socket updates

# Initialize components
print("Initializing components...")
face_detector = FaceDetector()
emotion_analyzer = EmotionAnalyzer()
depression_predictor = DepressionPredictor()

# Initialize video recommender if API key is available
youtube_api_key = os.getenv('YOUTUBE_API_KEY')
video_recommender = None
if youtube_api_key:
    try:
        print("Initializing video recommender...")
        video_recommender = VideoRecommender()
        print("Video recommender initialized successfully")
    except Exception as e:
        print(f"Error initializing video recommender: {str(e)}")
        print("Video recommendations will be disabled")

def get_webcam():
    """Get or initialize global webcam instance."""
    global global_cap
    
    with cap_lock:
        if global_cap is None or not global_cap.isOpened():
            print("Initializing webcam...")
            global_cap = init_webcam()
            
            if global_cap is None or not global_cap.isOpened():
                print("Could not connect to webcam")
                return None
                
        return global_cap

def process_frame():
    """Process a single frame from webcam."""
    global current_emotions, last_risk_check, last_socket_update
    
    try:
        cap = get_webcam()
        if cap is None:
            return None
            
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
            
        frame = resize_frame(frame)
        
        frame_with_faces, face_boxes = face_detector.detect_faces(frame, current_emotions)
        
        if face_boxes:
            largest_face = max(face_boxes, key=lambda box: box[2] * box[3])
            face_image = face_detector.extract_face(frame, largest_face)
            
            if face_image is not None:
                result = emotion_analyzer.analyze_emotion(face_image)
                
                if result:
                    current_emotions = result['emotions']
                    
                    # Throttle socket updates to once per second
                    current_time = time.time()
                    if current_time - last_socket_update >= 1.0:
                        socketio.emit('emotion_update', {
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'emotions': current_emotions
                        })
                        last_socket_update = current_time
                    
                    # Check depression risk periodically
                    current_time = datetime.now()
                    if (current_time - last_risk_check).total_seconds() >= risk_check_interval:
                        risk_score, is_high_risk = depression_predictor.predict_risk(
                            emotion_analyzer.emotion_history
                        )
                        
                        if is_high_risk and video_recommender:
                            recommendations = video_recommender.get_video_recommendations(
                                current_emotions
                            )
                            socketio.emit('recommendations', {
                                'videos': recommendations,
                                'risk_score': risk_score
                            })
                            
                        last_risk_check = current_time
                        save_session_data({
                            'timestamp': current_time.isoformat(),
                            'emotions': current_emotions,
                            'risk_score': risk_score,
                            'is_high_risk': is_high_risk
                        })
                    
        return frame_with_faces
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
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

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/history')
def history():
    """Show emotion history page."""
    return render_template('history.html')

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """API endpoint for video recommendations."""
    if not video_recommender:
        return jsonify({
            'error': 'Video recommendations are not available'
        }), 503
    
    try:
        emotion_data = request.json.get('emotion_data')
        if not emotion_data:
            return jsonify({'error': 'No emotion data provided'}), 400
            
        recommendations = video_recommender.get_video_recommendations(emotion_data)
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def handle_feedback():
    """Handle video recommendation feedback."""
    if not video_recommender:
        return jsonify({
            'error': 'Video recommendations are not available'
        }), 503
    
    try:
        data = request.json
        video_id = data.get('video_id')
        feedback_type = data.get('feedback_type')
        
        if not video_id:
            return jsonify({'error': 'No video ID provided'}), 400
            
        video_recommender.update_model(video_id, feedback_type)
        return jsonify({'status': 'success'})
        
    except Exception as e:
        print(f"Error handling feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    if current_emotions:
        emit('emotion_update', {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'emotions': current_emotions
        })

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
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        emotion_analyzer.load_history()
        
        print("Starting application at http://localhost:5000")
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=os.getenv('FLASK_DEBUG', '0') == '1'
        )
    finally:
        cleanup()