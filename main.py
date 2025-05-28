"""Main application for emotion health detection system."""

import os
import dotenv
dotenv.load_dotenv()

from flask import Flask, render_template, Response, jsonify, request, session
from flask_socketio import SocketIO
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
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
face_detector = FaceDetector()
emotion_analyzer = EmotionAnalyzer()
depression_predictor = DepressionPredictor()

# Initialize video recommender if API key is available
youtube_api_key = os.getenv('YOUTUBE_API_KEY')
video_recommender = None
if youtube_api_key:
    try:
        video_recommender = VideoRecommender(youtube_api_key)
    except Exception as e:
        print(f"Error initializing video recommender: {str(e)}")
        print("Video recommendations will be disabled")

# Global variables
global_cap = None  # Shared webcam instance
cap_lock = threading.Lock()
processing_frame = False
last_risk_check = datetime.now()
risk_check_interval = int(os.getenv('RISK_CHECK_INTERVAL', 300))
frame_lock = threading.Lock()
error_frame = None
current_emotions = None
emotion_update_interval = 0.5

def create_error_frame(message: str) -> np.ndarray:
    """Create an error message frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)
    
    (text_width, text_height), _ = cv2.getTextSize(
        message, font, font_scale, thickness
    )
    
    position = (
        (frame.shape[1] - text_width) // 2,
        (frame.shape[0] + text_height) // 2
    )
    
    cv2.putText(
        frame, message, position,
        font, font_scale, color, thickness
    )
    
    return frame

def get_webcam():
    """Get or initialize global webcam instance."""
    global global_cap
    
    with cap_lock:
        if global_cap is None or not global_cap.isOpened():
            print("Initializing webcam...")
            global_cap = init_webcam(
                width=int(os.getenv('WEBCAM_WIDTH', 640)),
                height=int(os.getenv('WEBCAM_HEIGHT', 480))
            )
            
            if global_cap is None or not global_cap.isOpened():
                print("Failed to initialize webcam")
                return None
                
        return global_cap

def get_frame():
    """Get a frame from webcam with emotion analysis."""
    global current_emotions, last_risk_check
    
    cap = get_webcam()
    if cap is None:
        return create_error_frame("Error: No webcam access")
            
    try:
        with frame_lock:
            ret, frame = cap.read()
            if not ret or frame is None:
                return create_error_frame("Error: No frame received")
                
            # Resize frame
            frame = resize_frame(frame)
            
            # Detect faces and analyze emotions
            frame_with_faces, face_boxes = face_detector.detect_faces(frame, current_emotions)
            
            if face_boxes:
                # Get largest face
                largest_face = max(face_boxes, key=lambda box: box[2] * box[3])
                face_image = face_detector.extract_face(frame, largest_face)
                
                if face_image is not None:
                    # Analyze emotions
                    result = emotion_analyzer.analyze_emotion(face_image)
                    
                    if result:
                        # Update current emotions
                        current_emotions = result['emotions']
                        
                        # Check depression risk periodically
                        current_time = datetime.now()
                        if (current_time - last_risk_check).total_seconds() >= risk_check_interval:
                            risk_score, is_high_risk = depression_predictor.predict_risk(
                                emotion_analyzer.emotion_history
                            )
                            
                            if is_high_risk and video_recommender:
                                # Get video recommendations if available
                                recommendations = video_recommender.get_collaborative_recommendations(
                                    result['dominant_emotion']
                                )
                                
                                # Emit recommendations to client
                                socketio.emit('recommendations', {
                                    'videos': recommendations,
                                    'risk_score': risk_score
                                })
                                
                            last_risk_check = current_time
                            
                            # Save session data
                            save_session_data({
                                'timestamp': current_time.isoformat(),
                                'emotions': result['emotions'],
                                'risk_score': risk_score,
                                'is_high_risk': is_high_risk
                            })
                
            return frame_with_faces
            
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return create_error_frame(f"Error: {str(e)}")

def generate_frames():
    """Generate processed video frames."""
    while True:
        frame = get_frame()
        if frame is not None:
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )

def emotion_update_thread():
    """Thread for updating emotion data via WebSocket."""
    last_update = 0
    while True:
        current_time = time.time()
        if current_emotions and (current_time - last_update) >= emotion_update_interval:
            timestamp = datetime.now().strftime('%H:%M:%S')
            socketio.emit('emotion_update', {
                'timestamp': timestamp,
                'emotions': current_emotions
            })
            last_update = current_time
        time.sleep(0.1)

@app.route('/')
def index():
    """Render main page."""
    return render_template(
        'index.html',
        video_recommendations_enabled=video_recommender is not None
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
    """Render history page."""
    return render_template('history.html')

@app.route('/update_preference', methods=['POST'])
def update_preference():
    """Update video preference."""
    if not video_recommender:
        return jsonify({
            'status': 'error',
            'message': 'Video recommendations are not enabled'
        })
        
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        liked = data.get('liked')
        
        if video_id is not None and liked is not None:
            video_recommender.update_preferences(video_id, liked)
            return jsonify({'status': 'success'})
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid data'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

def cleanup():
    """Clean up resources."""
    global global_cap
    with cap_lock:
        if global_cap is not None:
            global_cap.release()
            global_cap = None
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        # Ensure required directories exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Load emotion history if exists
        emotion_analyzer.load_history()
        
        # Start emotion update thread
        update_thread = threading.Thread(target=emotion_update_thread, daemon=True)
        update_thread.start()
        
        # Start the application
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=os.getenv('FLASK_DEBUG', '0') == '1'
        )
    finally:
        cleanup()