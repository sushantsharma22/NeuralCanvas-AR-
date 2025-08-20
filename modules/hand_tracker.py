"""
Advanced multi-hand tracking with enhanced features
"""
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class AdvancedHandTracker:
    def __init__(self, max_hands=2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Hand tracking enhancement
        self.hand_history = deque(maxlen=30)  # Track hand positions over time
        self.hand_velocities = {}
        self.hand_ids = {}  # Track individual hands
        self.next_hand_id = 0
        
        # Calibration and filtering
        self.calibration_data = None
        self.noise_filter = True
        self.stabilization_enabled = True
        
    def process_frame(self, frame):
        """Process frame and detect hands with enhancements"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        enhanced_results = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                # Get hand info
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                hand_score = handedness.classification[0].score
                
                # Apply noise filtering
                if self.noise_filter:
                    hand_landmarks = self._apply_noise_filter(hand_landmarks)
                
                # Calculate hand metrics
                hand_metrics = self._calculate_hand_metrics(hand_landmarks, frame.shape)
                
                # Track hand ID and velocity
                hand_id = self._assign_hand_id(hand_landmarks, hand_label)
                velocity = self._calculate_velocity(hand_id, hand_landmarks)
                
                enhanced_result = {
                    'landmarks': hand_landmarks,
                    'handedness': hand_label,
                    'confidence': hand_score,
                    'hand_id': hand_id,
                    'velocity': velocity,
                    'metrics': hand_metrics,
                    'filtered_landmarks': self._get_key_landmarks(hand_landmarks)
                }
                
                enhanced_results.append(enhanced_result)
        
        # Update hand history
        self.hand_history.append(enhanced_results)
        
        return enhanced_results
    
    def _apply_noise_filter(self, landmarks):
        """Apply noise filtering to landmarks"""
        # Simple moving average filter
        if len(self.hand_history) > 0:
            alpha = 0.7  # Smoothing factor
            
            # Get previous landmarks if available
            prev_results = self.hand_history[-1]
            if prev_results:
                prev_landmarks = prev_results[0]['landmarks']  # Assume first hand
                
                # Apply exponential moving average
                for i, (curr_lm, prev_lm) in enumerate(zip(landmarks.landmark, prev_landmarks.landmark)):
                    curr_lm.x = alpha * curr_lm.x + (1 - alpha) * prev_lm.x
                    curr_lm.y = alpha * curr_lm.y + (1 - alpha) * prev_lm.y
                    curr_lm.z = alpha * curr_lm.z + (1 - alpha) * prev_lm.z
        
        return landmarks
    
    def _calculate_hand_metrics(self, landmarks, frame_shape):
        """Calculate various hand metrics"""
        h, w = frame_shape[:2]
        
        # Convert landmarks to pixel coordinates
        points = []
        for lm in landmarks.landmark:
            points.append([int(lm.x * w), int(lm.y * h)])
        points = np.array(points)
        
        # Calculate hand bounding box
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # Hand size and center
        hand_width = x_max - x_min
        hand_height = y_max - y_min
        hand_center = [(x_min + x_max) // 2, (y_min + y_max) // 2]
        
        # Palm center (approximate)
        palm_landmarks = [0, 5, 9, 13, 17]  # Wrist and base of fingers
        palm_points = points[palm_landmarks]
        palm_center = np.mean(palm_points, axis=0).astype(int)
        
        # Finger lengths
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [2, 5, 9, 13, 17]
        finger_lengths = []
        
        for tip, base in zip(finger_tips, finger_bases):
            length = np.linalg.norm(points[tip] - points[base])
            finger_lengths.append(length)
        
        return {
            'bbox': (x_min, y_min, hand_width, hand_height),
            'center': hand_center,
            'palm_center': palm_center.tolist(),
            'hand_size': max(hand_width, hand_height),
            'finger_lengths': finger_lengths,
            'aspect_ratio': hand_width / hand_height if hand_height > 0 else 1.0
        }
    
    def _assign_hand_id(self, landmarks, handedness):
        """Assign consistent ID to tracked hands"""
        # Simple ID assignment based on position
        # In a production system, you'd use more sophisticated tracking
        
        palm_center = self._get_palm_center(landmarks)
        
        # Find closest existing hand
        min_distance = float('inf')
        closest_id = None
        
        for hand_id, prev_center in self.hand_ids.items():
            distance = np.linalg.norm(np.array(palm_center) - np.array(prev_center))
            if distance < min_distance and distance < 50:  # Threshold for same hand
                min_distance = distance
                closest_id = hand_id
        
        if closest_id is not None:
            # Update position
            self.hand_ids[closest_id] = palm_center
            return closest_id
        else:
            # New hand
            new_id = self.next_hand_id
            self.next_hand_id += 1
            self.hand_ids[new_id] = palm_center
            return new_id
    
    def _get_palm_center(self, landmarks):
        """Calculate palm center (normalized coordinates)"""
        idxs = [0, 5, 9, 13, 17]
        xs, ys = [], []
        for idx in idxs:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                xs.append(lm.x)
                ys.append(lm.y)

        if not xs:
            return [0.0, 0.0]

        return [sum(xs) / len(xs), sum(ys) / len(ys)]
    
    def _calculate_velocity(self, hand_id, landmarks):
        """Calculate hand velocity"""
        current_center = self._get_palm_center(landmarks)

        if hand_id not in self.hand_velocities:
            self.hand_velocities[hand_id] = deque(maxlen=5)
            # store initial position
            self.hand_velocities[hand_id].append(current_center)
            return [0.0, 0.0]

        prev_positions = self.hand_velocities[hand_id]
        if len(prev_positions) > 0:
            prev_center = prev_positions[-1]
            vx = current_center[0] - prev_center[0]
            vy = current_center[1] - prev_center[1]
            velocity = [vx, vy]
        else:
            velocity = [0.0, 0.0]

        # Add current position to history
        self.hand_velocities[hand_id].append(current_center)

        return velocity
    
    def _get_key_landmarks(self, landmarks):
        """Extract key landmarks for gesture recognition"""
        key_indices = [0, 4, 8, 12, 16, 20]  # Wrist, thumb tip, finger tips
        key_landmarks = {}
        
        for idx in key_indices:
            lm = landmarks.landmark[idx]
            key_landmarks[idx] = {'x': lm.x, 'y': lm.y, 'z': lm.z}
        
        return key_landmarks
    
    def draw_landmarks(self, frame, hand_results, enhanced=True):
        """Draw hand landmarks with enhancements"""
        for hand_data in hand_results:
            landmarks = hand_data['landmarks']
            hand_id = hand_data['hand_id']
            
            if enhanced:
                # Draw enhanced landmarks
                self._draw_enhanced_landmarks(frame, landmarks, hand_id)
            else:
                # Standard MediaPipe drawing
                self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
    
    def _draw_enhanced_landmarks(self, frame, landmarks, hand_id):
        """Draw enhanced hand landmarks with colors and effects"""
        h, w = frame.shape[:2]
        
        # Define colors for different fingers
        finger_colors = [
            (255, 0, 0),    # Thumb - Red
            (0, 255, 0),    # Index - Green
            (0, 0, 255),    # Middle - Blue
            (255, 255, 0),  # Ring - Cyan
            (255, 0, 255),  # Pinky - Magenta
        ]
        
        # Draw connections with colors
        connections = self.mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            
            start_x, start_y = int(start_point.x * w), int(start_point.y * h)
            end_x, end_y = int(end_point.x * w), int(end_point.y * h)
            
            # Determine finger for coloring
            finger_idx = self._get_finger_index(start_idx)
            color = finger_colors[finger_idx] if finger_idx >= 0 else (255, 255, 255)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
        
        # Draw landmark points
        for idx, landmark in enumerate(landmarks.landmark):
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            finger_idx = self._get_finger_index(idx)
            color = finger_colors[finger_idx] if finger_idx >= 0 else (255, 255, 255)
            
            # Different sizes for different types of landmarks
            if idx in [4, 8, 12, 16, 20]:  # Fingertips
                radius = 8
            elif idx == 0:  # Wrist
                radius = 10
            else:
                radius = 4
            
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 1)
        
        # Draw hand ID
        wrist = landmarks.landmark[0]
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        cv2.putText(frame, f'Hand {hand_id}', (wrist_x - 30, wrist_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _get_finger_index(self, landmark_idx):
        """Get finger index for a landmark"""
        if landmark_idx in [1, 2, 3, 4]:  # Thumb
            return 0
        elif landmark_idx in [5, 6, 7, 8]:  # Index
            return 1
        elif landmark_idx in [9, 10, 11, 12]:  # Middle
            return 2
        elif landmark_idx in [13, 14, 15, 16]:  # Ring
            return 3
        elif landmark_idx in [17, 18, 19, 20]:  # Pinky
            return 4
        else:
            return -1  # Wrist or other
    
    def get_hand_tracking_quality(self, hand_results):
        """Assess tracking quality"""
        if not hand_results:
            return {'quality': 'poor', 'score': 0.0, 'issues': ['no_hands_detected']}
        
        issues = []
        total_score = 0
        
        for hand_data in hand_results:
            confidence = hand_data['confidence']
            velocity_magnitude = np.linalg.norm(hand_data['velocity'])
            
            # Check confidence
            if confidence < 0.5:
                issues.append('low_confidence')
            
            # Check for excessive movement (shaking)
            if velocity_magnitude > 0.1:
                issues.append('excessive_movement')
            
            total_score += confidence
        
        avg_score = total_score / len(hand_results)
        
        if avg_score > 0.8 and not issues:
            quality = 'excellent'
        elif avg_score > 0.6:
            quality = 'good'
        elif avg_score > 0.4:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'quality': quality,
            'score': avg_score,
            'issues': issues,
            'num_hands': len(hand_results)
        }
