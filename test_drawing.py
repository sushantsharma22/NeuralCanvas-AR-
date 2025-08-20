#!/usr/bin/env python3
"""
Drawing engine test script to isolate drawing issues
"""
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from core.drawing_engine import NeuralDrawingEngine

def test_basic_drawing():
    """Test basic drawing functionality"""
    print("🧪 Testing basic drawing engine...")
    
    # Create a small test canvas
    width, height = 640, 480
    engine = NeuralDrawingEngine(width, height)
    
    print(f"Canvas size: {width}x{height}")
    print(f"Initial brush size: {engine.brush_size}")
    print(f"Initial color: {engine.current_color}")
    
    # Test single point
    print("\n1. Testing single point drawing...")
    engine.draw_point(100, 100, pressure=1.0)
    canvas = engine.get_composite_canvas()
    print(f"Stroke history length: {len(engine.stroke_history)}")
    
    # Test line drawing (simulate continuous stroke)
    print("\n2. Testing line drawing...")
    for i in range(20):
        x = 100 + i * 5
        y = 100 + i * 2
        pressure = 0.5 + 0.5 * (i / 20)  # Variable pressure
        engine.draw_point(x, y, pressure=pressure)
    
    print(f"Stroke history length after line: {len(engine.stroke_history)}")
    
    # Test different brush sizes
    print("\n3. Testing different brush sizes...")
    engine.change_brush_size(20)  # Thick brush
    for i in range(15):
        x = 200 + i * 4
        y = 200
        engine.draw_point(x, y, pressure=1.0)
    
    engine.change_brush_size(3)  # Thin brush
    for i in range(15):
        x = 200 + i * 4
        y = 250
        engine.draw_point(x, y, pressure=1.0)
    
    # Save test image
    final_canvas = engine.get_composite_canvas()
    test_file = "test_drawing_output.png"
    cv2.imwrite(test_file, final_canvas)
    print(f"\n✅ Test image saved: {test_file}")
    
    # Analyze results
    non_zero_pixels = np.count_nonzero(final_canvas)
    print(f"Non-zero pixels in canvas: {non_zero_pixels}")
    
    if non_zero_pixels > 0:
        print("✅ Drawing engine is working - pixels were drawn")
    else:
        print("❌ Drawing engine issue - no pixels drawn")
    
    return engine, final_canvas

def test_gesture_thresholds():
    """Test gesture recognition parameters"""
    print("\n🧪 Testing gesture recognition...")
    
    from core.gesture_engine import AdvancedGestureEngine
    gesture_engine = AdvancedGestureEngine()
    
    # Mock hand landmarks for testing
    class MockLandmark:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    # Create mock "draw" gesture (index finger extended)
    landmarks = []
    
    # Wrist
    landmarks.append(MockLandmark(0.5, 0.8, 0.0))  # 0: wrist
    
    # Thumb chain (folded)
    landmarks.append(MockLandmark(0.45, 0.75, 0.0))  # 1: thumb cmc
    landmarks.append(MockLandmark(0.42, 0.72, 0.0))  # 2: thumb mcp  
    landmarks.append(MockLandmark(0.40, 0.70, 0.0))  # 3: thumb ip
    landmarks.append(MockLandmark(0.38, 0.68, 0.0))  # 4: thumb tip (folded)
    
    # Index finger (extended)
    landmarks.append(MockLandmark(0.5, 0.7, 0.0))   # 5: index mcp
    landmarks.append(MockLandmark(0.5, 0.6, 0.0))   # 6: index pip
    landmarks.append(MockLandmark(0.5, 0.5, 0.0))   # 7: index dip
    landmarks.append(MockLandmark(0.5, 0.3, 0.0))   # 8: index tip (extended)
    
    # Middle finger (folded)
    landmarks.append(MockLandmark(0.55, 0.7, 0.0))  # 9: middle mcp
    landmarks.append(MockLandmark(0.55, 0.68, 0.0)) # 10: middle pip
    landmarks.append(MockLandmark(0.55, 0.66, 0.0)) # 11: middle dip
    landmarks.append(MockLandmark(0.55, 0.65, 0.0)) # 12: middle tip (folded)
    
    # Ring finger (folded)
    landmarks.append(MockLandmark(0.6, 0.7, 0.0))   # 13: ring mcp
    landmarks.append(MockLandmark(0.6, 0.68, 0.0))  # 14: ring pip
    landmarks.append(MockLandmark(0.6, 0.66, 0.0))  # 15: ring dip
    landmarks.append(MockLandmark(0.6, 0.65, 0.0))  # 16: ring tip (folded)
    
    # Pinky (folded)
    landmarks.append(MockLandmark(0.65, 0.7, 0.0))  # 17: pinky mcp
    landmarks.append(MockLandmark(0.65, 0.68, 0.0)) # 18: pinky pip
    landmarks.append(MockLandmark(0.65, 0.66, 0.0)) # 19: pinky dip
    landmarks.append(MockLandmark(0.65, 0.65, 0.0)) # 20: pinky tip (folded)
    
    # Test finger state detection
    finger_states = gesture_engine._get_finger_states(landmarks)
    print(f"Finger states (thumb, index, middle, ring, pinky): {finger_states}")
    print(f"Expected for DRAW: [0, 1, 0, 0, 0]")
    
    gesture, confidence = gesture_engine.recognize_gesture(landmarks)
    print(f"Mock gesture result: {gesture} (confidence: {confidence:.2f})")
    
    # Test all gesture patterns
    print("\n🧪 Testing all gesture patterns...")
    test_patterns = [
        ([0, 1, 0, 0, 0], "DRAW"),
        ([0, 1, 1, 0, 0], "NAVIGATE"), 
        ([1, 1, 1, 1, 1], "ERASE"),
        ([1, 1, 0, 0, 0], "COLOR_CHANGE")
    ]
    
    for pattern, expected in test_patterns:
        # Create landmarks matching the pattern
        test_landmarks = create_landmarks_for_pattern(pattern)
        states = gesture_engine._get_finger_states(test_landmarks)
        gesture, conf = gesture_engine.recognize_gesture(test_landmarks)
        print(f"Pattern {pattern} -> States: {states}, Gesture: {gesture} (expected: {expected})")
    
    return gesture_engine

def create_landmarks_for_pattern(finger_states):
    """Create mock landmarks matching finger state pattern"""
    class MockLandmark:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    landmarks = []
    # Wrist
    landmarks.append(MockLandmark(0.5, 0.8, 0.0))
    
    # Create landmarks based on finger states
    finger_positions = [
        # Thumb positions (extended vs folded)
        [(0.45, 0.75), (0.42, 0.72), (0.40, 0.70), (0.35, 0.65) if finger_states[0] else (0.38, 0.68)],
        # Index positions  
        [(0.5, 0.7), (0.5, 0.6), (0.5, 0.5), (0.5, 0.3) if finger_states[1] else (0.5, 0.65)],
        # Middle positions
        [(0.55, 0.7), (0.55, 0.6), (0.55, 0.5), (0.55, 0.3) if finger_states[2] else (0.55, 0.65)],
        # Ring positions
        [(0.6, 0.7), (0.6, 0.6), (0.6, 0.5), (0.6, 0.3) if finger_states[3] else (0.6, 0.65)],
        # Pinky positions
        [(0.65, 0.7), (0.65, 0.6), (0.65, 0.5), (0.65, 0.3) if finger_states[4] else (0.65, 0.65)]
    ]
    
    for finger_idx, positions in enumerate(finger_positions):
        for pos in positions:
            landmarks.append(MockLandmark(pos[0], pos[1], 0.0))
    
    return landmarks

if __name__ == "__main__":
    print("🎨 NeuralCanvas Drawing System Test")
    print("=" * 50)
    
    try:
        # Test drawing engine
        engine, canvas = test_basic_drawing()
        
        # Test gesture recognition
        gesture_engine = test_gesture_thresholds()
        
        print("\n" + "=" * 50)
        print("🎯 Test Summary:")
        print("- Drawing engine initialized successfully")
        print("- Check test_drawing_output.png for visual results")
        print("- Run main app to test full integration")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
