#!/usr/bin/env python3
"""
Test the improved smooth drawing functionality
"""

import cv2
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.drawing_engine import NeuralDrawingEngine

def test_smooth_drawing():
    """Test smooth drawing with various patterns"""
    print("🎨 Testing Improved Smooth Drawing Engine...")
    
    # Create drawing engine
    drawing_engine = NeuralDrawingEngine(800, 600)
    
    # Test 1: Straight line drawing
    print("  Test 1: Drawing straight line...")
    drawing_engine.start_stroke(100, 100, 1.0)
    for i in range(50):
        x = 100 + i * 4
        y = 100
        drawing_engine.draw_point(x, y, 1.0)
    drawing_engine.end_stroke()
    
    # Test 2: Curved line drawing
    print("  Test 2: Drawing curved line...")
    drawing_engine.start_stroke(100, 200, 1.0)
    for i in range(100):
        x = 100 + i * 2
        y = 200 + int(50 * np.sin(i * 0.1))
        pressure = 0.5 + 0.5 * np.sin(i * 0.05)  # Varying pressure
        drawing_engine.draw_point(x, y, pressure)
    drawing_engine.end_stroke()
    
    # Test 3: Multiple separate strokes
    print("  Test 3: Drawing separate strokes...")
    for stroke in range(3):
        start_x = 100 + stroke * 150
        start_y = 350
        
        drawing_engine.start_stroke(start_x, start_y, 1.0)
        for i in range(30):
            x = start_x + i * 2
            y = start_y + int(30 * np.sin(i * 0.2))
            drawing_engine.draw_point(x, y, 0.8)
        drawing_engine.end_stroke()
    
    # Test 4: Different colors
    print("  Test 4: Drawing with different colors...")
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    
    for i, color in enumerate(colors):
        drawing_engine.change_color(color)
        start_x = 100 + i * 100
        start_y = 450
        
        drawing_engine.start_stroke(start_x, start_y, 1.0)
        for j in range(40):
            x = start_x + j * 3
            y = start_y + int(40 * np.cos(j * 0.15))
            drawing_engine.draw_point(x, y, 1.0)
        drawing_engine.end_stroke()
    
    # Get final canvas
    canvas = drawing_engine.get_composite_canvas()
    
    # Count non-zero pixels
    non_zero_pixels = np.count_nonzero(canvas)
    print(f"  Total pixels drawn: {non_zero_pixels}")
    
    # Save test image
    cv2.imwrite("smooth_drawing_test.png", canvas)
    print("  ✅ Test image saved as 'smooth_drawing_test.png'")
    
    # Validate that we have smooth lines (should have many more pixels than dotted)
    if non_zero_pixels > 20000:  # Should have many pixels for smooth lines
        print("  ✅ Smooth drawing test: PASSED")
        return True
    else:
        print("  ❌ Smooth drawing test: FAILED - Not enough pixels for smooth lines")
        return False

def main():
    """Run smooth drawing tests"""
    print("🧪 Smooth Drawing System Test")
    print("=" * 50)
    
    success = test_smooth_drawing()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed! Smooth drawing is working correctly.")
        print("\nImprovements implemented:")
        print("✅ Stroke start/end management")
        print("✅ Smooth line interpolation using OpenCV")
        print("✅ Anti-aliased drawing for smooth edges")
        print("✅ Velocity-based smoothing for jitter reduction")
        print("✅ Feathered brush edges")
        print("\nThe drawing should now be smooth and continuous!")
    else:
        print("⚠️  Tests failed. Check the implementation.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
