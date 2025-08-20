#!/usr/bin/env python3
"""
Quick gesture test to confirm all actions work
"""
import cv2
import time
from config.settings import Config
from core.gesture_engine import AdvancedGestureEngine
from modules.hand_tracker import AdvancedHandTracker

def quick_gesture_test():
    print("🚀 Quick Gesture Test")
    print("="*40)
    print("Test each gesture briefly:")
    print("👍 Thumb + Index = COLOR_CHANGE")
    # Shape mode, voice activate and emotion removed
    print("✌️🖕 Three fingers = SAVE")
    print("🖖 Four fingers = CLEAR")
    print("Press 'q' to quit")
    print("="*40)
    
    gesture_engine = AdvancedGestureEngine()
    hand_tracker = AdvancedHandTracker(max_hands=2)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed")
        return
    
    gesture_cooldowns = {}
    cooldown_time = 1.0  # 1 second cooldown for this test
    
    # Track what we've tested
    tested_gestures = set()
    target_gestures = {"COLOR_CHANGE", "SAVE", "CLEAR"}
    
    frame_count = 0
    
    while len(tested_gestures) < len(target_gestures):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        hand_results = hand_tracker.process_frame(frame)
        
        if hand_results:
            for hand_data in hand_results:
                landmarks = hand_data['landmarks']
                hand_id = hand_data['hand_id']
                
                gesture, confidence = gesture_engine.recognize_gesture(landmarks.landmark)
                finger_states = gesture_engine._get_finger_states(landmarks.landmark)
                
                if gesture in target_gestures and confidence > 0.3:
                    cooldown_key = (hand_id, gesture)
                    now = time.time()
                    last_time = gesture_cooldowns.get(cooldown_key, 0)
                    
                    if now - last_time > cooldown_time:
                        gesture_cooldowns[cooldown_key] = now
                        tested_gestures.add(gesture)
                        
                        print(f"✅ {gesture}: Fingers {finger_states} -> WOULD EXECUTE ACTION")
                        
                        if gesture == "COLOR_CHANGE":
                            print("   → Would cycle to next color")
                        elif gesture == "SAVE":
                            print("   → Would save artwork")
                        elif gesture == "CLEAR":
                            print("   → Would clear canvas")
                        
                        remaining = target_gestures - tested_gestures
                        if remaining:
                            print(f"📋 Still need to test: {remaining}")
                        else:
                            print("🎉 ALL GESTURES TESTED SUCCESSFULLY!")
        
        # Show frame
        cv2.imshow('Quick Gesture Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(tested_gestures) == len(target_gestures):
        print("\n🎉 SUCCESS: All gestures work!")
    else:
        missing = target_gestures - tested_gestures
        print(f"\n⚠️  Missing gestures: {missing}")

if __name__ == "__main__":
    quick_gesture_test()
