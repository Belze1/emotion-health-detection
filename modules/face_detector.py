"""Face detection module."""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional

class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        """Initialize face detector.
        
        Args:
            min_detection_confidence: Minimum confidence value for face detection
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        
    def draw_emotion_text(self, frame: np.ndarray, text: str, pos: tuple, 
                         font_scale: float = 0.6, thickness: int = 2) -> None:
        """Draw text with background on frame.
        
        Args:
            frame: Image to draw on
            text: Text to draw
            pos: Position (x, y) to draw text
            font_scale: Font scale
            thickness: Line thickness
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw background
        x, y = pos
        cv2.rectangle(
            frame,
            (x, y - text_height - 5),
            (x + text_width + 5, y + baseline),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
    def detect_faces(self, frame: np.ndarray, emotions: dict = None) -> Tuple[np.ndarray, list]:
        """Detect faces in frame.
        
        Args:
            frame: Input image frame
            emotions: Optional dictionary of emotions with probabilities
            
        Returns:
            Tuple containing:
                - Frame with face detections drawn
                - List of face bounding boxes
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        
        # Process the frame
        results = self.face_detection.process(rgb_frame)
        
        # Draw detection results
        annotated_frame = frame.copy()
        face_boxes = []
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame_width)
                y = int(bbox.ymin * frame_height)
                w = int(bbox.width * frame_width)
                h = int(bbox.height * frame_height)
                
                # Store face box coordinates
                face_boxes.append((x, y, w, h))
                
                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )
                
                # Draw emotion information if available
                if emotions:
                    # Sort emotions by probability
                    sorted_emotions = sorted(
                        emotions.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Get top 2 emotions
                    top_emotions = sorted_emotions[:2]
                    
                    # Draw each emotion on separate lines
                    y_offset = y - 10  # Start above face box
                    for emotion, prob in top_emotions:
                        text = f"{emotion}: {int(prob * 100)}%"
                        self.draw_emotion_text(
                            annotated_frame,
                            text,
                            (x, y_offset)
                        )
                        y_offset -= 25  # Move up for next emotion
        
        return annotated_frame, face_boxes
    
    def extract_face(self, frame: np.ndarray, face_box: tuple) -> Optional[np.ndarray]:
        """Extract face region from frame.
        
        Args:
            frame: Input image frame
            face_box: Tuple of (x, y, w, h) coordinates
            
        Returns:
            Face region as numpy array or None if extraction fails
        """
        try:
            x, y, w, h = face_box
            
            # Add padding (20%)
            padding = 0.2
            x_pad = int(w * padding)
            y_pad = int(h * padding)
            
            x1 = max(0, x - x_pad)
            y1 = max(0, y - y_pad)
            x2 = min(frame.shape[1], x + w + x_pad)
            y2 = min(frame.shape[0], y + h + y_pad)
            
            face_region = frame[y1:y2, x1:x2]
            return face_region
            
        except Exception as e:
            print(f"Lỗi trích xuất khuôn mặt: {str(e)}")
            return None

    def release(self):
        """Release resources."""
        self.face_detection.close()