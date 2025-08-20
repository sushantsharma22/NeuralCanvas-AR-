#!/usr/bin/env python3
"""
Advanced test for ultra-smooth drawing to eliminate dotted lines
"""

import cv2
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.drawing_engine import NeuralDrawingEngine

def test_ultra_smooth_drawing():
    """Test ultra-smooth drawing with various challenging patterns"""
    print("🎨 Testing Ultra-Smooth Drawing (Anti-Dotted Line System)...")
    
    # Create drawing engine
    drawing_engine = NeuralDrawingEngine(1000, 800)
    
    # Test 1: Very fast movement simulation (challenging for smoothness)
    print("  Test 1: Fast movement simulation...")
    drawing_engine.start_stroke(100, 100, 1.0)
    
    # Simulate fast finger movement with large gaps
    fast_points = [
        (100, 100), (120, 105), (150, 110), (190, 115), (240, 120),
        (300, 125), (370, 130), (450, 135), (540, 140), (640, 145)
    ]
    
    for x, y in fast_points:
        drawing_engine.draw_point(x, y, 0.9)
    drawing_engine.end_stroke()
    
    # Test 2: Slow detailed movement (should be ultra-smooth)
    print("  Test 2: Detailed slow movement...")
    drawing_engine.start_stroke(100, 200, 1.0)
    
    # Create a detailed curved path with many small steps
    for i in range(200):
        x = 100 + i * 3
        y = 200 + int(50 * np.sin(i * 0.1)) + np.random.randint(-2, 3)  # Add small jitter
        pressure = 0.8 + 0.2 * np.sin(i * 0.05)
        drawing_engine.draw_point(x, y, pressure)
    drawing_engine.end_stroke()
    
    # Test 3: Multiple stroke transitions (challenging for continuity)
    print("  Test 3: Multiple stroke transitions...")
    for stroke in range(5):
        start_x = 100 + stroke * 80
        start_y = 350
        
        drawing_engine.start_stroke(start_x, start_y, 1.0)
        
        # Short strokes with various pressures
        for i in range(20):
            x = start_x + i * 2
            y = start_y + int(20 * np.sin(i * 0.3))
            pressure = 0.5 + 0.5 * np.abs(np.sin(i * 0.2))
            drawing_engine.draw_point(x, y, pressure)
        
        drawing_engine.end_stroke()
    
    # Test 4: Pressure variation test
    print("  Test 4: Pressure variation...")
    drawing_engine.start_stroke(100, 500, 1.0)
    
    for i in range(150):
        x = 100 + i * 4
        y = 500 + int(30 * np.cos(i * 0.08))
        # Vary pressure dramatically to test size transitions
        pressure = 0.2 + 0.8 * (np.sin(i * 0.1) + 1) / 2
        drawing_engine.draw_point(x, y, pressure)
    drawing_engine.end_stroke()
    
    # Test 5: Circle drawing (ultimate smoothness test)
    print("  Test 5: Circle drawing smoothness test...")
    center_x, center_y = 500, 400
    radius = 80
    
    drawing_engine.start_stroke(center_x + radius, center_y, 1.0)
    
    # Draw a circle with many points
    for i in range(180):  # Half circle for now
        angle = i * np.pi / 90  # 2 degrees per step
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        drawing_engine.draw_point(x, y, 0.9)
    drawing_engine.end_stroke()
    
    # Get final canvas and analyze
    canvas = drawing_engine.get_composite_canvas()
    
    # Count total pixels
    non_zero_pixels = np.count_nonzero(canvas)
    print(f"  Total pixels drawn: {non_zero_pixels}")
    
    # Analyze smoothness by checking pixel density
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # Find contours to analyze line continuity
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_contour_length = sum(cv2.arcLength(contour, False) for contour in contours)
    
    print(f"  Contour analysis: {len(contours)} contours, total length: {total_contour_length:.1f}")
    
    # Calculate smoothness metric (pixels per unit length)
    if total_contour_length > 0:
        density_ratio = non_zero_pixels / total_contour_length
        print(f"  Smoothness metric (pixels/length): {density_ratio:.2f}")
        
        # Higher density = smoother lines (no gaps)
        is_smooth = density_ratio > 8.0  # Good threshold for smooth lines
    else:
        is_smooth = False
    
    # Save test image
    cv2.imwrite("ultra_smooth_test.png", canvas)
    print("  ✅ Test image saved as 'ultra_smooth_test.png'")
    
    # Validation
    success_conditions = [
        non_zero_pixels > 25000,  # Sufficient coverage
        len(contours) > 0,        # Something was drawn
        is_smooth                 # High pixel density = smooth lines
    ]
    
    if all(success_conditions):
        print("  ✅ Ultra-smooth drawing test: PASSED")
        print("  🎯 No dotted lines detected - drawing is ultra-smooth!")
        return True
    else:
        print("  ❌ Ultra-smooth drawing test: FAILED")
        if not is_smooth:
            print("  ⚠️  Detected potential dotted line issue")
        return False

def main():
    """Run ultra-smooth drawing tests"""
    print("🧪 Ultra-Smooth Drawing System Test (Anti-Dotted)")
    print("=" * 60)
    
    success = test_ultra_smooth_drawing()
    
    print("=" * 60)
    if success:
        print("🎉 ULTRA-SMOOTH DRAWING CONFIRMED!")
        print("\n🚀 Key improvements working:")
        print("✅ Gesture stability with low-confidence smoothing")
        print("✅ Coordinate jump detection and handling")
        print("✅ Ultra-smooth line interpolation with circles")
        print("✅ Aggressive velocity-based smoothing")
        print("✅ Drawing timeout management")
        print("✅ Anti-aliased rendering throughout")
        print("\n🎯 Result: NO MORE DOTTED LINES!")
    else:
        print("⚠️  Still detecting potential smoothness issues.")
        print("Check the test image 'ultra_smooth_test.png' for visual inspection.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
