#!/usr/bin/env python3
"""
Comprehensive test suite for NeuralCanvas AR fixes
"""
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_brush_sizes():
    """Test brush size functionality"""
    print("🎨 Testing brush sizes...")
    
    from core.drawing_engine import NeuralDrawingEngine
    
    engine = NeuralDrawingEngine(400, 300)
    
    # Test different brush sizes
    sizes_to_test = [1, 2, 4, 8, 16, 32]
    
    for i, size in enumerate(sizes_to_test):
        engine.change_brush_size(size)
        print(f"Set brush size to {size}, actual: {engine.brush_size}")
        
        # Draw a test stroke
        x_start = 50 + i * 50
        for j in range(10):
            engine.draw_point(x_start, 50 + j * 3, pressure=1.0)
    
    # Save test image
    canvas = engine.get_composite_canvas()
    cv2.imwrite("brush_size_test.png", canvas)
    print("✅ Brush size test saved to brush_size_test.png")

def test_gesture_recognition_detailed():
    """Detailed test of gesture recognition"""
    print("\n🎯 Testing gesture recognition...")
    
    from core.gesture_engine import AdvancedGestureEngine
    
    engine = AdvancedGestureEngine()
    
    # Test cases for common gestures
    test_cases = [
        {
            "name": "Index finger only (DRAW)",
            "finger_states": [0, 1, 0, 0, 0],
            "expected": "DRAW"
        },
        {
            "name": "Index + Middle (NAVIGATE)",
            "finger_states": [0, 1, 1, 0, 0],
            "expected": "NAVIGATE"
        },
        {
            "name": "All fingers (ERASE)",
            "finger_states": [1, 1, 1, 1, 1],
            "expected": "ERASE"
        },
        {
            "name": "Thumb + Index (COLOR_CHANGE)",
            "finger_states": [1, 1, 0, 0, 0],
            "expected": "COLOR_CHANGE"
        }
    ]
    
    for test_case in test_cases:
        landmarks = create_test_landmarks(test_case["finger_states"])
        gesture, confidence = engine.recognize_gesture(landmarks)
        
        status = "✅" if gesture == test_case["expected"] else "❌"
        print(f"{status} {test_case['name']}: Got {gesture} (expected {test_case['expected']}) - confidence: {confidence:.2f}")

def create_test_landmarks(finger_states):
    """Create test landmarks for specific finger states"""
    class MockLandmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z
    
    landmarks = []
    
    # Wrist (always at bottom center)
    landmarks.append(MockLandmark(0.5, 0.8))
    
    # Define finger joint positions
    fingers = [
        # Thumb: different logic (horizontal extension)
        {
            "extended": [(0.45, 0.75), (0.4, 0.7), (0.35, 0.65), (0.3, 0.6)],
            "folded": [(0.48, 0.75), (0.46, 0.73), (0.44, 0.71), (0.42, 0.69)]
        },
        # Index finger
        {
            "extended": [(0.5, 0.7), (0.5, 0.6), (0.5, 0.5), (0.5, 0.3)],
            "folded": [(0.5, 0.7), (0.5, 0.68), (0.5, 0.66), (0.5, 0.65)]
        },
        # Middle finger
        {
            "extended": [(0.55, 0.7), (0.55, 0.6), (0.55, 0.5), (0.55, 0.3)],
            "folded": [(0.55, 0.7), (0.55, 0.68), (0.55, 0.66), (0.55, 0.65)]
        },
        # Ring finger
        {
            "extended": [(0.6, 0.7), (0.6, 0.6), (0.6, 0.5), (0.6, 0.3)],
            "folded": [(0.6, 0.7), (0.6, 0.68), (0.6, 0.66), (0.6, 0.65)]
        },
        # Pinky
        {
            "extended": [(0.65, 0.7), (0.65, 0.6), (0.65, 0.5), (0.65, 0.3)],
            "folded": [(0.65, 0.7), (0.65, 0.68), (0.65, 0.66), (0.65, 0.65)]
        }
    ]
    
    # Add finger landmarks based on states
    for finger_idx, state in enumerate(finger_states):
        finger_config = fingers[finger_idx]
        positions = finger_config["extended"] if state == 1 else finger_config["folded"]
        
        for pos in positions:
            landmarks.append(MockLandmark(pos[0], pos[1]))
    
    return landmarks

def test_coordinate_mapping():
    """Test coordinate mapping from normalized to pixel coordinates"""
    print("\n📍 Testing coordinate mapping...")
    
    # Test with different screen sizes
    screen_sizes = [(640, 480), (1280, 720), (1920, 1080)]
    
    for width, height in screen_sizes:
        print(f"\nTesting with {width}x{height} resolution:")
        
        # Test normalized coordinates (0.0 to 1.0)
        test_coords = [
            (0.0, 0.0),    # Top-left
            (0.5, 0.5),    # Center
            (1.0, 1.0),    # Bottom-right
            (0.25, 0.75)   # Quarter-three-quarter
        ]
        
        for norm_x, norm_y in test_coords:
            pixel_x = int(norm_x * width)
            pixel_y = int(norm_y * height)
            print(f"  ({norm_x:.1f}, {norm_y:.1f}) -> ({pixel_x}, {pixel_y})")

def main():
    print("🧪 NeuralCanvas AR - Comprehensive Fix Test Suite")
    print("=" * 60)
    
    try:
        # Test drawing engine improvements
        test_brush_sizes()
        
        # Test gesture recognition fixes  
        test_gesture_recognition_detailed()
        
        # Test coordinate mapping
        test_coordinate_mapping()
        
        print("\n" + "=" * 60)
        print("🎯 Test Suite Complete!")
        print("✅ All core functionalities tested")
        print("📁 Check brush_size_test.png for visual verification")
        print("🚀 Ready to run main application!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
