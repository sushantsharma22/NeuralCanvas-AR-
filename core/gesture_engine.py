"""
Advanced gesture recognition engine with AI prediction
"""
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from collections import deque
import pickle
import os
from config.settings import Config

class AdvancedGestureEngine:
    def __init__(self):
        self.gesture_history = deque(maxlen=50)
        self.velocity_history = deque(maxlen=10)
        self.acceleration_history = deque(maxlen=5)
        self.gesture_classifier = self._load_or_create_classifier()
        self.prediction_confidence = 0.0
        self.gesture_stability_counter = 0
        self.prev_landmarks = None
        
    def _load_or_create_classifier(self):
        """Load pre-trained classifier or create new one"""
        model_path = 'assets/models/gesture_classifier.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Create and train a basic classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            # In a real implementation, you'd train this with actual data
            return clf
    
    def extract_advanced_features(self, landmarks):
        """Extract comprehensive features from hand landmarks"""
        if not landmarks:
            return np.zeros(84)  # Feature vector size
            
        features = []
        
        # Basic landmark positions (21 * 2 = 42 features)
        for landmark in landmarks:
            features.extend([landmark.x, landmark.y])
        
        # Finger angles and distances (21 features)
        for i in range(21):
            if i < 20:
                # Angle between consecutive landmarks
                v1 = np.array([landmarks[i].x, landmarks[i].y])
                v2 = np.array([landmarks[i+1].x, landmarks[i+1].y])
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
                features.append(angle)
            else:
                features.append(0)
        
        # Hand orientation and size (21 features)
        # Use a few stable landmarks to estimate palm center (wrist + bases)
        idxs = [0, 2, 5, 9, 13]
        palm_landmarks = [landmarks[i] for i in idxs if i < len(landmarks)]
        palm_center = np.mean([[lm.x, lm.y] for lm in palm_landmarks], axis=0)

        for landmark in landmarks:
            dx = landmark.x - palm_center[0]
            dy = landmark.y - palm_center[1]
            distance = np.sqrt(dx * dx + dy * dy)
            features.append(distance)
        
        return np.array(features[:84])  # Ensure consistent size
    
    def calculate_velocity_acceleration(self, current_landmarks):
        """Calculate velocity and acceleration of hand movement"""
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return np.zeros(3), np.zeros(3)
        
        # Calculate velocity (change in position)
        current_pos = np.array([current_landmarks[4].x, current_landmarks[4].y, current_landmarks[4].z])
        prev_pos = np.array([self.prev_landmarks[4].x, self.prev_landmarks[4].y, self.prev_landmarks[4].z])
        velocity = current_pos - prev_pos
        
        self.velocity_history.append(velocity)
        
        # Calculate acceleration (change in velocity)
        if len(self.velocity_history) >= 2:
            acceleration = self.velocity_history[-1] - self.velocity_history[-2]
            self.acceleration_history.append(acceleration)
        else:
            acceleration = np.zeros(3)
        
        self.prev_landmarks = current_landmarks
        return velocity, acceleration
    
    def recognize_gesture(self, landmarks):
        """Advanced gesture recognition with prediction"""
        if not landmarks:
            return "NONE", 0.0
        
        # Extract features
        features = self.extract_advanced_features(landmarks)
        velocity, acceleration = self.calculate_velocity_acceleration(landmarks)
        
        # Combine features with motion data
        motion_features = np.concatenate([velocity, acceleration])
        full_features = np.concatenate([features, motion_features]).reshape(1, -1)
        
        # Basic gesture recognition (enhanced version)
        gesture = self._recognize_basic_gesture(landmarks)
        confidence = self._calculate_gesture_confidence(landmarks, gesture)
        
        # Add to history for stability
        self.gesture_history.append((gesture, confidence))
        
        # Stabilize gesture recognition
        stable_gesture = self._stabilize_gesture()
        
        return stable_gesture, confidence
    
    def _recognize_basic_gesture(self, landmarks):
        """Enhanced basic gesture recognition with better patterns"""
        finger_states = self._get_finger_states(landmarks)
        
        # Count extended fingers
        extended_count = sum(finger_states)
        
        # Debug output (uncomment for debugging)
        # print(f"Finger states: {finger_states}, Count: {extended_count}")
        
        # CRITICAL: Closed hand detection (all fingers down) - STOP DRAWING
        if finger_states == [0, 0, 0, 0, 0]:
            return "IDLE"  # Closed fist = completely idle, no drawing
        # Try matching against configured gesture patterns using simple voting
        # This helps when a single finger (often the thumb) is noisy and
        # avoids swapping SAVE <-> CLEAR incorrectly.
        # Explicit rules for SAVE (three fingers: index+middle+ring) and
        # CLEAR (four fingers) to reduce rapid toggling caused by thumb/pinky noise.
        # Treat any 4-extended-finger case as CLEAR to make the action robust.
        if extended_count == 4:
            return "CLEAR"
        # Prefer SAVE when index+middle+ring are extended; keep this rule strict
        # to avoid confusing 2-finger NAVIGATE with SAVE.
        if extended_count == 3 and finger_states[1] == 1 and finger_states[2] == 1 and finger_states[3] == 1:
            return "SAVE"

        # Try matching against configured gesture patterns using simple voting
        best_match = None
        best_score = -1
        for name, pattern in Config.GESTURES.items():
            matches = sum(1 for a, b in zip(finger_states, pattern) if a == b)
            if matches > best_score:
                best_score = matches
                best_match = name

        # If we have a strong match (4 or 5 matching fingers), accept it
        if best_score >= 4:
            return best_match

        # Otherwise fall back to original heuristics for common cases
        if finger_states == [0, 1, 0, 0, 0]:
            return "DRAW"
        if extended_count == 1 and finger_states[1] == 1:
            return "DRAW"
        if extended_count == 2 and finger_states[1] == 1 and finger_states[2] == 1:
            return "NAVIGATE"
        if extended_count == 5:
            return "ERASE"

        return "NONE"
    
    def _get_finger_states(self, landmarks):
        """Get finger extension states with reliable detection"""
        if not landmarks or len(landmarks) < 21:
            return [0, 0, 0, 0, 0]
            
        states = []
        
        # Thumb (landmark 4) - check if it's away from the palm
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3] 
        thumb_mcp = landmarks[2]
        
        # Thumb is extended if tip is farther from palm center than IP joint
        palm_center_x = (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5
        
        thumb_tip_dist = abs(thumb_tip.x - palm_center_x)
        thumb_ip_dist = abs(thumb_ip.x - palm_center_x)
        
        thumb_extended = thumb_tip_dist > thumb_ip_dist * 1.2
        states.append(1 if thumb_extended else 0)
        
        # Index finger (landmarks 5,6,7,8)
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        
        # Finger is extended if tip is significantly higher (lower y) than joints
        index_extended = (index_tip.y < index_pip.y - 0.03) and (index_tip.y < index_mcp.y - 0.05)
        states.append(1 if index_extended else 0)
        
        # Middle finger (landmarks 9,10,11,12) 
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        middle_mcp = landmarks[9]
        
        middle_extended = (middle_tip.y < middle_pip.y - 0.03) and (middle_tip.y < middle_mcp.y - 0.05)
        states.append(1 if middle_extended else 0)
        
        # Ring finger (landmarks 13,14,15,16)
        ring_tip = landmarks[16] 
        ring_pip = landmarks[14]
        ring_mcp = landmarks[13]
        
        ring_extended = (ring_tip.y < ring_pip.y - 0.03) and (ring_tip.y < ring_mcp.y - 0.05)
        states.append(1 if ring_extended else 0)
        
        # Pinky (landmarks 17,18,19,20)
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18] 
        pinky_mcp = landmarks[17]
        # Make pinky detection slightly stricter to avoid noise
        pinky_extended = ((pinky_tip.y < pinky_pip.y - 0.035) and 
                          (pinky_tip.y < pinky_mcp.y - 0.06) and
                          (abs(pinky_tip.x - pinky_mcp.x) > 0.01))
        states.append(1 if pinky_extended else 0)

        return states

    def _calculate_gesture_confidence(self, landmarks, gesture):
        """Calculate confidence score for gesture recognition"""
        if gesture == "NONE" or gesture == "IDLE":
            return 1.0 if gesture == "IDLE" else 0.0  # High confidence for closed hand
        
        # Base confidence on finger positioning accuracy
        finger_states = self._get_finger_states(landmarks)
        expected_states = {
            "DRAW": [0, 1, 0, 0, 0],
            "NAVIGATE": [0, 1, 1, 0, 0],
            "ERASE": [1, 1, 1, 1, 1],
            "COLOR_CHANGE": [1, 1, 0, 0, 0],
            "SAVE": [0, 1, 1, 1, 0],
            "CLEAR": [1, 1, 1, 1, 0],
            "IDLE": [0, 0, 0, 0, 0]
        }
        
        if gesture in expected_states:
            expected = expected_states[gesture]
            matches = sum(1 for i, j in zip(finger_states, expected) if i == j)
            confidence = matches / len(expected)
        else:
            confidence = 0.5

        # Boost confidence for explicit 3/4-finger detections to reduce flicker
        extended_count = sum(finger_states)
        if gesture == "CLEAR" and extended_count == 4:
            confidence = max(confidence, 0.95)
        if gesture == "SAVE" and extended_count == 3:
            confidence = max(confidence, 0.95)
        
        return confidence
    
    def _stabilize_gesture(self):
        """Stabilize gesture recognition over time"""
        if len(self.gesture_history) < 1:
            return "NONE"
        
        # Get recent gestures
        recent_gestures = [g[0] for g in list(self.gesture_history)[-3:]]
        
        # Find most common gesture
        from collections import Counter
        gesture_counts = Counter(recent_gestures)
        most_common = gesture_counts.most_common(1)
        
        # For SAVE/CLEAR, require short-history agreement (at least 2 of last 3)
        # or extremely high confidence to avoid rapid flicker between the two.
        if most_common:
            latest_gesture, latest_confidence = self.gesture_history[-1]
            if latest_gesture in {"SAVE", "CLEAR"}:
                if latest_confidence > 0.9:
                    return latest_gesture
                elif most_common[0][1] >= 2:
                    return most_common[0][0]
            else:
                # For other gestures, allow an immediate high-confidence response
                if latest_confidence > 0.85:
                    return latest_gesture
                elif most_common[0][1] >= 2:  # At least 2 occurrences
                    return most_common[0][0]
        
        return "NONE"
    
    def predict_next_position(self, current_landmarks):
        """Predict next hand position using motion history"""
        if len(self.velocity_history) < 3:
            return None
        
        # Simple linear prediction based on velocity
        current_pos = np.array([current_landmarks[4].x, current_landmarks[4].y])
        avg_velocity = np.mean([v[:2] for v in list(self.velocity_history)[-3:]], axis=0)
        
        predicted_pos = current_pos + avg_velocity * 2  # Predict 2 frames ahead
        
        return predicted_pos
