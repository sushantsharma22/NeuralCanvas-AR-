"""
Facial expression analysis for emotion-based color changing
"""
import cv2
import mediapipe as mp
import numpy as np

class EmotionColorMapper:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Emotion to color mapping
        self.emotion_colors = {
            'happy': (0, 255, 255),      # Yellow
            'sad': (255, 0, 0),          # Blue
            'angry': (0, 0, 255),        # Red
            'surprised': (255, 0, 255),  # Magenta
            'neutral': (255, 255, 255),  # White
            'focused': (0, 255, 0),      # Green
            'excited': (0, 165, 255)     # Orange
        }
        
        self.current_emotion = 'neutral'
        self.emotion_history = []
        
    def analyze_emotion(self, frame):
        """Analyze facial expression and return corresponding emotion"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            emotion = self._classify_emotion(face_landmarks, frame.shape)
            if emotion:
                self.emotion_history.append(emotion)

            # Stabilize emotion detection
            if len(self.emotion_history) > 10:
                self.emotion_history.pop(0)

            # Get most frequent emotion
            from collections import Counter
            emotion_counts = Counter(self.emotion_history) if self.emotion_history else Counter(['neutral'])
            stable_emotion = emotion_counts.most_common(1)[0][0]

            self.current_emotion = stable_emotion
            return stable_emotion, self.emotion_colors.get(stable_emotion, self.emotion_colors['neutral'])
        
        return 'neutral', self.emotion_colors['neutral']

    def _classify_emotion(self, face_landmarks, frame_shape):
        """Simple heuristic-based emotion classification from face landmarks.

        This is a lightweight fallback so the app can run without a trained model.
        It uses mouth openness and eyebrow/eye distances as rough signals.
        """
        try:
            # Convert normalized landmarks to pixel coords for a few key indices
            h, w = frame_shape[:2]
            lm = face_landmarks.landmark

            # Mouth: upper lip (13), lower lip (14) - approximate indices
            top_lip = lm[13]
            bottom_lip = lm[14]
            mouth_open = (bottom_lip.y - top_lip.y) * h

            # Eyebrow vs eye distance (use a few landmark indices common in face mesh)
            left_eye = lm[33]
            left_eyebrow = lm[70]
            eye_brow_gap = (left_eyebrow.y - left_eye.y) * h

            # Simple rules
            if mouth_open > 25:
                return 'surprised'
            if mouth_open > 8 and mouth_open <= 25:
                return 'happy'
            if eye_brow_gap < -5:
                return 'angry'

            return 'neutral'
        except Exception:
            return 'neutral'
