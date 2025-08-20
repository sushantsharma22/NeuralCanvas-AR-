#!/usr/bin/env python3
"""
Real-time gesture debugging script
"""
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from core.gesture_engine import AdvancedGestureEngine
from modules.hand_tracker import AdvancedHandTracker

def debug_gesture_detection():
    """Debug gesture detection with real camera input"""
    print("🔍 Starting real-time gesture debugging...")
    print("Press 'q' to quit, 's' to save debug info")
    
    # Initialize components
    hand_tracker = AdvancedHandTracker(max_hands=1)
    gesture_engine = AdvancedGestureEngine()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)  # Mirror image
        
        # Process hand tracking
        hand_results = hand_tracker.process_frame(frame)
        
        if hand_results:
            for hand_data in hand_results:
                landmarks = hand_data['landmarks']
                
                # Get finger states for debugging
                finger_states = gesture_engine._get_finger_states(landmarks.landmark)
                gesture, confidence = gesture_engine.recognize_gesture(landmarks.landmark)
                
                # Draw hand landmarks
                hand_tracker.draw_landmarks(frame, hand_results, enhanced=True)
                
                # Display debug info
                y_offset = 30
                cv2.putText(frame, f"Finger States: {finger_states}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                y_offset += 30
                cv2.putText(frame, f"Gesture: {gesture} ({confidence:.2f})", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show expected patterns
                y_offset += 50
                patterns = [
                    "DRAW: [0,1,0,0,0] - Index only",
                    "NAVIGATE: [0,1,1,0,0] - Index+Middle", 
                    "ERASE: [1,1,1,1,1] - All fingers",
                    "COLOR: [1,1,0,0,0] - Thumb+Index"
                ]
                
                for i, pattern in enumerate(patterns):
                    cv2.putText(frame, pattern, (10, y_offset + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow('Gesture Debug', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('debug_frame.png', frame)
            print("📸 Debug frame saved")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    debug_gesture_detection()
