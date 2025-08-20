#!/usr/bin/env python3
"""
Debug script to test gesture action execution
"""
import cv2
import time
from collections import deque

# Import custom modules
from config.settings import Config
from core.gesture_engine import AdvancedGestureEngine
from modules.hand_tracker import AdvancedHandTracker

def test_gesture_actions():
    print("🔍 Testing Gesture Action Execution")
    print("="*50)
    print("This will test if gesture actions execute properly:")
    print("👍 Thumb + Index: COLOR_CHANGE")
    print("🤟 Index + Pinky: SHAPE_MODE") 
    # Voice and shape features removed
    print("✌️🖕 Three fingers: SAVE")
    print("🖖 Four fingers: CLEAR")
    print("✊ Closed fist: IDLE")
    print("="*50)
    
    # Initialize components
    gesture_engine = AdvancedGestureEngine()
    hand_tracker = AdvancedHandTracker(max_hands=2)
    
    # Mock cooldowns like main.py
    gesture_cooldowns = {}
    gesture_cooldown_seconds = 0.6
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("\n🎥 Camera initialized. Show gestures to test actions!")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Process hands
        hand_results = hand_tracker.process_frame(frame)
        
        if hand_results:
            for hand_data in hand_results:
                landmarks = hand_data['landmarks']
                hand_id = hand_data['hand_id']
                
                # Get gesture
                gesture, confidence = gesture_engine.recognize_gesture(landmarks.landmark)
                finger_states = gesture_engine._get_finger_states(landmarks.landmark)
                
                # Show finger states for debugging
                if gesture != "NONE":
                    print(f"Frame {frame_count:4d}: {gesture:12s} (conf: {confidence:.2f}) Fingers: {finger_states}")
                
                # Test quick actions with same logic as main.py
                quick_actions = {"COLOR_CHANGE", "SAVE", "CLEAR"}
                now = time.time()
                
                if gesture in quick_actions and confidence > 0.3:
                    cooldown_key = (hand_id, gesture)
                    last_ts = gesture_cooldowns.get(cooldown_key, 0)
                    
                    if now - last_ts > gesture_cooldown_seconds:
                        # Execute the action
                        result = execute_test_action(gesture)
                        if result:
                            print(f"⚡ EXECUTED: {gesture} -> {result}")
                            gesture_cooldowns[cooldown_key] = now
                        else:
                            print(f"❌ FAILED: {gesture} -> No result")
                    else:
                        remaining = gesture_cooldown_seconds - (now - last_ts)
                        print(f"⏳ COOLDOWN: {gesture} (wait {remaining:.1f}s)")
                
                elif gesture == "IDLE":
                    print(f"✊ IDLE detected - would stop drawing")
                
                elif gesture in ["DRAW", "ERASE"]:
                    print(f"✏️  {gesture} detected - would draw/erase")
        
        # Show frame
        cv2.imshow('Gesture Action Debug', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Gesture action test completed!")

def execute_test_action(gesture):
    """Mock action execution to test logic"""
    if gesture == "COLOR_CHANGE":
        colors = list(Config.COLORS.keys())
        # Simulate cycling through colors
        current_idx = 0  # Would normally track current color
        next_color = colors[(current_idx + 1) % len(colors)]
        return f"Changed color to {next_color}"
    
    # Shape and voice features removed
    
    elif gesture == "SAVE":
        # Simulate save
        return "Artwork saved"
    
    elif gesture == "CLEAR":
        # Simulate clear
        return "Canvas cleared"
    
    return None

if __name__ == "__main__":
    test_gesture_actions()
