#!/usr/bin/env python3
"""
Test closed hand detection to verify drawing stops properly
"""

import cv2
import mediapipe as mp
import numpy as np
from core.gesture_engine import AdvancedGestureEngine

def main():
    print("🧪 Testing Closed Hand Detection")
    print("=" * 50)
    print("This test will help verify that:")
    print("1. ✋ Open hand gestures are detected properly")
    print("2. ✊ Closed hand (IDLE) stops all drawing")
    print("3. 👆 Index finger drawing works correctly")
    print("4. No unwanted drawing continuation occurs")
    print("=" * 50)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Initialize gesture engine
    gesture_engine = AdvancedGestureEngine()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n🎥 Camera initialized. Instructions:")
    print("👆 Point index finger = DRAW gesture")
    print("✌️  Index + Middle finger = NAVIGATE gesture") 
    print("✊ Close hand completely = IDLE (should STOP drawing)")
    print("🛑 Press 'q' to quit")
    print()
    
    frame_count = 0
    last_gesture = None
    consecutive_idle = 0
    consecutive_draw = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = hands.process(rgb_frame)
        
        current_gesture = "NO_HAND"
        confidence = 0.0
        finger_states = [0, 0, 0, 0, 0]
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get gesture
                gesture, conf = gesture_engine.recognize_gesture(hand_landmarks.landmark)
                current_gesture = gesture
                confidence = conf
                finger_states = gesture_engine._get_finger_states(hand_landmarks.landmark)
                
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                break
        
        # Track gesture consistency
        if current_gesture == last_gesture:
            if current_gesture == "IDLE":
                consecutive_idle += 1
            elif current_gesture == "DRAW":
                consecutive_draw += 1
        else:
            consecutive_idle = 0
            consecutive_draw = 0
        
        # Display information
        status_color = (0, 255, 0)  # Green for good
        if current_gesture == "IDLE":
            status_color = (0, 0, 255)  # Red for idle
        elif current_gesture == "DRAW":
            status_color = (255, 0, 0)  # Blue for draw
        
        # Display current state
        cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Fingers: {finger_states}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show consistency counters
        cv2.putText(frame, f"Consecutive IDLE: {consecutive_idle}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Consecutive DRAW: {consecutive_draw}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Key test indicators
        if current_gesture == "IDLE" and consecutive_idle > 5:
            cv2.putText(frame, "✅ CLOSED HAND DETECTED - DRAWING WOULD STOP", 
                       (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif current_gesture == "DRAW" and finger_states == [0, 1, 0, 0, 0]:
            cv2.putText(frame, "✅ PERFECT DRAW GESTURE - DRAWING ACTIVE", 
                       (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif current_gesture in ["NAVIGATE", "ERASE", "COLOR_CHANGE"]:
            cv2.putText(frame, f"✅ {current_gesture} GESTURE - NO DRAWING", 
                       (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Closed Hand Detection Test', frame)
        
        # Print important state changes
        if current_gesture != last_gesture and last_gesture is not None:
            print(f"Frame {frame_count:4d}: {last_gesture:>12} → {current_gesture:<12} (conf: {confidence:.2f}) Fingers: {finger_states}")
            
            if current_gesture == "IDLE":
                print("                🛑 CLOSED HAND - DRAWING SHOULD STOP!")
            elif current_gesture == "DRAW" and last_gesture != "DRAW":
                print("                ✏️  DRAWING GESTURE - DRAWING SHOULD START!")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        last_gesture = current_gesture
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Closed hand detection test completed!")
    print("Key behaviors to verify:")
    print("1. Closed fist (✊) should show 'IDLE' gesture consistently")
    print("2. Index finger pointing (👆) should show 'DRAW' gesture")
    print("3. Gesture transitions should be clear and stable")

if __name__ == "__main__":
    main()
