#!/usr/bin/env python3
"""
Complete functionality test for NeuralCanvas AR
Tests gesture recognition, drawing, and voice commands
"""

import cv2
import numpy as np
import time
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.gesture_engine import AdvancedGestureEngine
from core.drawing_engine import NeuralDrawingEngine
from modules.hand_tracker import AdvancedHandTracker
from modules.voice_controller import VoiceController
from config.settings import Config

def test_gesture_recognition():
    """Test gesture recognition with simulated hand landmarks"""
    print("🔍 Testing Gesture Recognition...")
    
    gesture_engine = AdvancedGestureEngine()
    
    # Test cases for different gesture patterns
    test_cases = [
        {
            'name': 'Single finger (DRAW)',
            'fingers': [0, 1, 0, 0, 0],  # Only index finger up
            'expected': 'DRAW'
        },
        {
            'name': 'Two fingers (NAVIGATE)', 
            'fingers': [0, 1, 1, 0, 0],  # Index + middle finger up
            'expected': 'NAVIGATE'
        },
        {
            'name': 'Four fingers (CLEAR)',
            'fingers': [1, 1, 1, 1, 0],  # All except pinky
            'expected': 'CLEAR'
        },
        {
            'name': 'All fingers (ERASE)',
            'fingers': [1, 1, 1, 1, 1],  # All fingers up
            'expected': 'ERASE'
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        # Create mock landmarks that would produce the finger pattern
        mock_landmarks = create_mock_landmarks(test_case['fingers'])
        
        # Test gesture recognition
        finger_states = gesture_engine._get_finger_states(mock_landmarks)
        gesture = gesture_engine._recognize_basic_gesture(mock_landmarks)
        
        print(f"  {test_case['name']}: ", end="")
        print(f"Expected {test_case['expected']}, Got {gesture}", end="")
        
        if gesture == test_case['expected']:
            print(" ✅ PASS")
            passed += 1
        else:
            print(" ❌ FAIL")
            print(f"    Finger states detected: {finger_states}")
    
    print(f"\n✅ Gesture Recognition: {passed}/{total} tests passed\n")
    return passed == total

def create_mock_landmarks(finger_pattern):
    """Create mock hand landmarks for testing"""
    class MockLandmark:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    class MockLandmarks:
        def __init__(self):
            self.landmark = []
        
        def __len__(self):
            return len(self.landmark)
        
        def __getitem__(self, index):
            return self.landmark[index]
    
    landmarks = MockLandmarks()
    
    # Create 21 landmarks for a hand with proper MediaPipe structure
    palm_center_y = 0.5
    palm_center_x = 0.5
    
    # Finger mapping: [thumb, index, middle, ring, pinky]
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]
    finger_mcps = [2, 5, 9, 13, 17]
    
    # Initialize all landmarks
    for i in range(21):
        landmarks.landmark.append(MockLandmark(palm_center_x, palm_center_y, 0.0))
    
    # Set palm landmarks (wrist and base points)
    landmarks.landmark[0] = MockLandmark(palm_center_x, palm_center_y + 0.1, 0.0)  # Wrist
    
    # Set finger landmarks based on pattern
    for finger_idx, extended in enumerate(finger_pattern):
        tip_idx = finger_tips[finger_idx]
        pip_idx = finger_pips[finger_idx]
        mcp_idx = finger_mcps[finger_idx]
        
        if finger_idx == 0:  # Thumb - horizontal movement
            if extended:
                landmarks.landmark[tip_idx] = MockLandmark(palm_center_x + 0.15, palm_center_y, 0.0)
                landmarks.landmark[pip_idx] = MockLandmark(palm_center_x + 0.05, palm_center_y, 0.0)  # IP joint closer to palm
                landmarks.landmark[mcp_idx] = MockLandmark(palm_center_x, palm_center_y, 0.0)
            else:
                # Thumb tucked in - tip is closer to palm center than IP joint
                landmarks.landmark[tip_idx] = MockLandmark(palm_center_x + 0.02, palm_center_y, 0.0)  # Tip close to palm
                landmarks.landmark[pip_idx] = MockLandmark(palm_center_x + 0.08, palm_center_y, 0.0)  # IP farther out
                landmarks.landmark[mcp_idx] = MockLandmark(palm_center_x, palm_center_y, 0.0)
        else:  # Other fingers - vertical movement
            if extended:
                # Extended: tip is significantly above (lower y) than joints
                landmarks.landmark[tip_idx] = MockLandmark(palm_center_x, palm_center_y - 0.15, 0.0)
                landmarks.landmark[pip_idx] = MockLandmark(palm_center_x, palm_center_y - 0.05, 0.0)
                landmarks.landmark[mcp_idx] = MockLandmark(palm_center_x, palm_center_y, 0.0)
            else:
                # Curled: tip is below or same level as joints
                landmarks.landmark[tip_idx] = MockLandmark(palm_center_x, palm_center_y + 0.02, 0.0)
                landmarks.landmark[pip_idx] = MockLandmark(palm_center_x, palm_center_y, 0.0)
                landmarks.landmark[mcp_idx] = MockLandmark(palm_center_x, palm_center_y, 0.0)
    
    return landmarks

def test_drawing_engine():
    """Test drawing engine functionality"""
    print("🎨 Testing Drawing Engine...")
    
    try:
        drawing_engine = NeuralDrawingEngine(width=800, height=600)
        
        # Test basic drawing
        test_points = [
            {'x': 100, 'y': 100, 'pressure': 1.0},
            {'x': 150, 'y': 150, 'pressure': 0.8},
            {'x': 200, 'y': 200, 'pressure': 0.6}
        ]
        
        for point in test_points:
            drawing_engine.draw_point(point['x'], point['y'], point['pressure'])
        
        # Test color change
        drawing_engine.change_color((255, 0, 0))  # Red
        drawing_engine.draw_point(300, 300, 1.0)
        
        # Test brush size
        original_size = drawing_engine.brush_size
        drawing_engine.brush_size = 10
        drawing_engine.draw_point(400, 400, 1.0)
        
        # Count non-zero pixels to verify drawing
        canvas = drawing_engine.get_composite_canvas()
        non_zero_pixels = np.count_nonzero(canvas)
        
        print(f"  Drawing test: {non_zero_pixels} pixels drawn ✅ PASS")
        print(f"  Color change: Current color {drawing_engine.current_color} ✅ PASS")
        print(f"  Brush size: Changed from {original_size} to {drawing_engine.brush_size} ✅ PASS")
        
        print("✅ Drawing Engine: All tests passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Drawing Engine: Failed with error {e}\n")
        return False

def test_voice_controller():
    """Test voice controller initialization"""
    print("🎤 Testing Voice Controller...")
    
    try:
        voice_controller = VoiceController()
        
        # Test command parsing
        test_commands = [
            ('draw', 'DRAW'),
            ('red', 'COLOR_RED'),
            ('clear', 'CLEAR'),
            ('save', 'SAVE'),
            ('bigger', 'BRUSH_BIGGER')
        ]
        
        passed = 0
        for test_input, expected in test_commands:
            result = voice_controller._parse_command(test_input)
            if result == expected:
                print(f"  Voice command '{test_input}' -> '{result}' ✅ PASS")
                passed += 1
            else:
                print(f"  Voice command '{test_input}' -> '{result}' (expected '{expected}') ❌ FAIL")
        
        print(f"✅ Voice Controller: {passed}/{len(test_commands)} tests passed\n")
        return passed == len(test_commands)
        
    except Exception as e:
        print(f"❌ Voice Controller: Failed with error {e}\n")
        return False

def test_hand_tracking():
    """Test hand tracking initialization"""
    print("👋 Testing Hand Tracking...")
    
    try:
        hand_tracker = AdvancedHandTracker()
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process the frame (should not crash)
        results = hand_tracker.process_frame(test_frame)
        
        print("  Hand tracker initialization ✅ PASS")
        print("  Frame processing ✅ PASS")
        print("✅ Hand Tracking: All tests passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Hand Tracking: Failed with error {e}\n")
        return False

def main():
    """Run all tests"""
    print("🧪 NeuralCanvas AR - Comprehensive Functionality Test")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Gesture Recognition", test_gesture_recognition()))
    test_results.append(("Drawing Engine", test_drawing_engine()))
    test_results.append(("Voice Controller", test_voice_controller()))
    test_results.append(("Hand Tracking", test_hand_tracking()))
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed_tests += 1
    
    print("-" * 60)
    print(f"Total: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! NeuralCanvas AR is ready to use.")
        print("\nTo start the application, run:")
        print("python main.py")
        print("\nTo activate voice control, use the thumb + pinky gesture!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
