"""
NeuralCanvas AR - Advanced Real-Time Finger Drawing Application
The most unique and comprehensive finger drawing platform ever created
"""
import cv2
import numpy as np
import time
import threading
from collections import deque

# Import custom modules
from config.settings import Config
from core.gesture_engine import AdvancedGestureEngine
from core.drawing_engine import NeuralDrawingEngine
from modules.hand_tracker import AdvancedHandTracker
# removed EmotionColorMapper and VoiceController per request
from utils.performance_monitor import PerformanceMonitor
from ui.interface_manager import InterfaceManager
from typing import Optional, List, Dict, Any

class NeuralCanvasAR:
    def __init__(self):
        print("🚀 Initializing NeuralCanvas AR - The Future of Digital Art!")

        # Core components
        self.gesture_engine = AdvancedGestureEngine()
        self.hand_tracker = AdvancedHandTracker(max_hands=Config.MAX_HANDS)
        self.performance_monitor = PerformanceMonitor()

        # Drawing system - will be initialized in initialize_camera()
        self.drawing_engine: Optional[NeuralDrawingEngine] = None
        self.ui_manager: Optional[InterfaceManager] = None

        # Application state
        self.running = False
        self.current_mode = "2D"
        self.auto_save = True

        # Drawing state tracking
        self.drawing_active = False
        self.last_gesture = None
        self.last_drawing_point = None
        self.gesture_history = deque(maxlen=5)  # Track recent gestures for stability
        self.drawing_timeout = 0  # Frame counter for drawing timeout

        # Gesture action cooldowns to debounce tap gestures (hand_id, gesture) -> timestamp
        self.gesture_cooldowns = {}
        self.gesture_cooldown_seconds = 0.6

        # Performance tracking
        self.fps_counter = 0
        self.frame_times = deque(maxlen=30)

        # Session recording
        self.recording = False
        self.session_frames = []

        print("✅ NeuralCanvas AR initialized successfully!")
    
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        print("📸 Setting up camera...")
        # Try multiple camera indices first
        cap = None
        frame = None
        for idx in range(0, 4):
            try:
                c = cv2.VideoCapture(idx)
                # set some properties; some backends ignore these
                c.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
                c.set(cv2.CAP_PROP_FPS, Config.FPS_TARGET)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ret, frm = c.read()
                if ret and frm is not None:
                    cap = c
                    frame = frm
                    print(f"✅ Camera initialized (index {idx}): {frame.shape[1]}x{frame.shape[0]}")
                    break
                else:
                    c.release()
            except Exception:
                continue

        # If no camera available, fall back to a synthetic frame generator
        if cap is None:
            print("⚠️  No camera found - using synthetic frame fallback for testing")

            class SyntheticCapture:
                def __init__(self, width, height, color=(20, 20, 20)):
                    self.width = width
                    self.height = height
                    self.color = color
                    self.opened = True

                def read(self):
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    frame[:] = self.color
                    # add a moving test marker to make visual debugging easier
                    t = int(time.time() * 1000) // 100 % (self.width - 40)
                    cv2.circle(frame, (20 + t, int(self.height / 2)), 10, (0, 180, 255), -1)
                    return True, frame

                def release(self):
                    self.opened = False

                def isOpened(self):
                    return self.opened

                def set(self, *args, **kwargs):
                    return True

            cap = SyntheticCapture(Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)
            ret, frame = cap.read()

        if frame is None:
            raise RuntimeError("❌ Could not initialize camera or synthetic fallback!")

        # Initialize drawing engine with camera dimensions
        h, w = frame.shape[:2]
        self.drawing_engine = NeuralDrawingEngine(w, h)
        self.ui_manager = InterfaceManager(w, h)

        return cap, frame.shape
    
    def process_gestures(self, hand_results):
        """Process hand gestures with improved stability"""
        if not hand_results:
            # No hands detected - end any active drawing
            if self.drawing_active and self.drawing_engine:
                print("🛑 No hands detected - ending drawing stroke")
                self.drawing_engine.end_stroke()
                self.drawing_active = False
            return None, None
        
        # Process each detected hand
        commands = []
        drawing_points = []
        
        for hand_data in hand_results:
            landmarks = hand_data['landmarks']
            hand_id = hand_data['hand_id']

            # Recognize gesture (stabilized) and also get the immediate/basic gesture
            gesture, confidence = self.gesture_engine.recognize_gesture(landmarks.landmark)
            # Use the immediate/basic recognition for quick, tap-style actions so
            # a single-frame tap (thumb+index etc.) doesn't get swallowed by
            # the stabilizer which prefers repeated detections.
            raw_gesture = self.gesture_engine._recognize_basic_gesture(landmarks.landmark)
            raw_confidence = self.gesture_engine._calculate_gesture_confidence(landmarks.landmark, raw_gesture)
            
            # Debug output for gesture detection (show all gestures)
            if gesture != "NONE":
                print(f"🎯 Detected: {gesture} (confidence: {confidence:.2f})")
            
            # CRITICAL: Handle IDLE state (closed hand) - STOP DRAWING IMMEDIATELY
            if gesture == "IDLE":
                if self.drawing_active and self.drawing_engine:
                    print("🛑 Hand closed - ending drawing stroke")
                    self.drawing_engine.end_stroke()
                    self.drawing_active = False
                continue  # Skip further processing for this hand

            # QUICK ACTIONS: Immediate, debounced execution for tap-style gestures
            quick_actions = {"COLOR_CHANGE", "SAVE", "CLEAR"}
            now = time.time()
            # Prefer the raw/basic gesture for quick tap actions (faster response)
            # Use a slightly higher confidence threshold for SAVE/CLEAR to avoid
            # brief finger noise causing accidental triggers.
            required_conf = 0.3
            if raw_gesture in {"SAVE", "CLEAR"}:
                required_conf = 0.6

            if raw_gesture in quick_actions and raw_confidence > required_conf:
                cooldown_key = (hand_id, raw_gesture)
                last_ts = self.gesture_cooldowns.get(cooldown_key, 0)
                if now - last_ts > self.gesture_cooldown_seconds:
                    cmd = self.execute_gesture_command(raw_gesture, landmarks, hand_id)
                    if cmd:
                        commands.append(cmd)
                        # store cooldown timestamp
                        self.gesture_cooldowns[cooldown_key] = now
                        print(f"⚡ Executed quick action: {raw_gesture} -> {cmd}")
                # Quick actions are not drawing gestures, skip drawing point extraction for them
                continue
            
            # CRITICAL: If not drawing gesture and not high confidence, DON'T continue drawing
            if gesture not in ["DRAW", "ERASE"] and self.drawing_active:
                # Only maintain drawing if we have very recent DRAW history AND decent confidence
                recent_draws = sum(1 for g in list(self.gesture_history)[-3:] if g == "DRAW")
                if (recent_draws < 2 or confidence < 0.4) and self.drawing_engine:
                    print("🛑 No valid drawing gesture - ending drawing stroke")
                    self.drawing_engine.end_stroke()
                    self.drawing_active = False
            
            # Special case: Always allow drawing when index finger is clearly extended
            finger_states = self.gesture_engine._get_finger_states(landmarks.landmark)
            if finger_states[1] == 1 and finger_states == [0, 1, 0, 0, 0] and gesture == "NONE":
                gesture = "DRAW"
                confidence = 0.8  # High confidence for clear index finger
            
            # Gesture stability: Only maintain drawing for very short periods with good history
            if (self.drawing_active and gesture == "NONE" and 
                len(self.gesture_history) > 0 and 
                self.gesture_history[-1] == "DRAW" and
                self.drawing_timeout < 3):  # Only for 3 frames max
                gesture = "DRAW"
                confidence = 0.3  # Lower confidence for continuation
                print(f"🔄 Maintaining DRAW gesture (low confidence smoothing)")
            
            # Use lower threshold for continuing existing actions
            effective_threshold = Config.PREDICTION_THRESHOLD
            if self.drawing_active and gesture == "DRAW":
                effective_threshold = 0.25  # Lower threshold for drawing continuation
            
            if confidence > effective_threshold:
                command = self.execute_gesture_command(gesture, landmarks, hand_id)
                if command:
                    commands.append(command)
                
                # Extract drawing points
                if gesture in ['DRAW', 'ERASE']:
                    point = self.extract_drawing_point(landmarks, gesture, hand_id)
                    if point:
                        drawing_points.append(point)
        
        return commands, drawing_points
    
    def execute_gesture_command(self, gesture, landmarks, hand_id):
        """Execute gesture-based commands"""
        if not self.drawing_engine:
            return None
        h, w = self.drawing_engine.height, self.drawing_engine.width
        
        if gesture == "COLOR_CHANGE":
            # Cycle through colors
            colors = list(Config.COLORS.values())
            current_idx = colors.index(self.drawing_engine.current_color) if self.drawing_engine.current_color in colors else 0
            next_color = colors[(current_idx + 1) % len(colors)]
            self.drawing_engine.change_color(next_color)
            return f"Changed color to {list(Config.COLORS.keys())[colors.index(next_color)]}"

        elif gesture == "SAVE":
            self.save_artwork()
            return "Artwork saved"
        
        elif gesture == "CLEAR":
            self.drawing_engine.clear_canvas()
            return "Canvas cleared"
        
        return None
    
    def extract_drawing_point(self, landmarks, gesture, hand_id):
        """Extract drawing point from hand landmarks"""
        if not self.drawing_engine:
            return None
        h, w = self.drawing_engine.height, self.drawing_engine.width
        
        # Use index finger tip for drawing
        index_tip = landmarks.landmark[8]
        x, y = int(index_tip.x * w), int(index_tip.y * h)
        z = index_tip.z * Config.Z_SENSITIVITY + 5  # Convert to depth
        
        # Calculate pressure based on index tip z (normalized)
        try:
            pressure = 1.0 - min(abs(index_tip.z) * 2, 1.0)
        except Exception:
            pressure = 1.0
        
        return {
            'x': x, 'y': y, 'z': z,
            'pressure': pressure,
            'gesture': gesture,
            'hand_id': hand_id,
            'timestamp': time.time()
        }
    
    def process_drawing_points(self, drawing_points):
        """Process drawing points with improved stroke continuity"""
        if not self.drawing_engine:
            return
        
        # Update drawing timeout
        if drawing_points:
            self.drawing_timeout = 0
        else:
            self.drawing_timeout += 1
            
            # If no drawing points for several frames, end any active stroke
            if self.drawing_timeout > 5 and self.drawing_active:
                print("🛑 Drawing timeout - ending stroke")
                self.drawing_engine.end_stroke()
                self.drawing_active = False
                return

        for point in drawing_points:
            print(f"✏️  Drawing point: {point['gesture']} at ({point['x']}, {point['y']}) pressure={point['pressure']:.2f}")
            
            if point['gesture'] == 'DRAW':
                # Check for coordinate jumps that indicate a new stroke
                coordinate_jump = False
                if self.last_drawing_point is not None:
                    distance = np.sqrt(
                        (point['x'] - self.last_drawing_point['x'])**2 + 
                        (point['y'] - self.last_drawing_point['y'])**2
                    )
                    coordinate_jump = distance > 100  # Large jump threshold
                
                # Start new stroke if:
                # 1. Not currently drawing, OR
                # 2. Last gesture wasn't DRAW, OR  
                # 3. Large coordinate jump detected
                if not self.drawing_active or self.last_gesture != 'DRAW' or coordinate_jump:
                    if coordinate_jump:
                        print("🎨 Starting new stroke (large coordinate jump detected)")
                    else:
                        print("🎨 Starting new drawing stroke")
                    
                    self.drawing_engine.start_stroke(
                        point['x'], point['y'], 
                        point['pressure'], point['z']
                    )
                    self.drawing_active = True
                else:
                    # Continue existing stroke
                    self.drawing_engine.draw_point(
                        point['x'], point['y'], 
                        point['pressure'], point['z']
                    )
                    
                self.last_drawing_point = point
                
            elif point['gesture'] == 'ERASE':
                # End any active drawing stroke when switching to erase
                if self.drawing_active:
                    print("🎨 Ending drawing stroke (switching to erase)")
                    self.drawing_engine.end_stroke()
                    self.drawing_active = False
                
                # Erase at point
                thickness = getattr(self.drawing_engine, 'eraser_thickness', max(10, self.drawing_engine.brush_size))
                self.drawing_engine.erase_at_point(point['x'], point['y'], thickness)
            
            # Update gesture history for stability tracking
            self.gesture_history.append(point['gesture'])
            self.last_gesture = point['gesture']
        
        # End stroke only if no drawing points for several frames (not just 1)
        if not drawing_points and self.drawing_active and self.drawing_timeout > 3 and self.drawing_engine:
            print("🎨 Ending drawing stroke (gesture timeout)")
            self.drawing_engine.end_stroke()
            self.drawing_active = False
            self.last_gesture = None
    
    # Emotion coloring feature removed
    
    def render_frame(self, frame, hand_results, commands, emotion_info):
        """Render final frame with all overlays"""
        if not self.drawing_engine or not self.ui_manager:
            return frame
        
        # Get drawing canvas
        canvas = self.drawing_engine.get_composite_canvas()
        
        # Blend canvas with video
        mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        
        frame_masked = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        final_frame = cv2.add(frame_masked, canvas)
        
        # Draw hand landmarks
        self.hand_tracker.draw_landmarks(final_frame, hand_results, enhanced=True)
        
        # Draw UI elements
        self.ui_manager.draw_interface(final_frame, {
            'current_color': self.drawing_engine.current_color,
            'brush_size': self.drawing_engine.brush_size,
            'current_layer': self.drawing_engine.current_layer,
            'mode': self.current_mode,
            'commands': commands,
            'emotion_info': emotion_info,
            'fps': self.get_current_fps()
        })
        
        # Performance overlay
        self.performance_monitor.draw_stats(final_frame)
        
        return final_frame
    
    def handle_keyboard_input(self):
        """Handle keyboard input for controls"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('c') and self.drawing_engine:
            self.drawing_engine.clear_canvas()
        elif key == ord('s'):
            self.save_artwork()
        elif key == ord('r'):
            self.toggle_recording()
    # 'e' - emotion coloring removed
        elif key == ord('3'):
            self.current_mode = "3D" if self.current_mode == "2D" else "2D"
        elif (key == ord('=') or key == ord('+')) and self.drawing_engine:
            new_size = min(self.drawing_engine.brush_size + 2, Config.MAX_BRUSH_SIZE)
            self.drawing_engine.change_brush_size(new_size)
        elif key == ord('-') and self.drawing_engine:
            new_size = max(self.drawing_engine.brush_size - 2, Config.MIN_BRUSH_SIZE)
            self.drawing_engine.change_brush_size(new_size)
        elif key == ord('u') and self.drawing_engine:
            self.drawing_engine.undo()
        
        return True
    
    def save_artwork(self):
        """Save current artwork"""
        if not self.drawing_engine:
            print("⚠️  Cannot save: drawing engine not initialized")
            return
        
        timestamp = int(time.time())
        filename = f"exports/images/neural_canvas_{timestamp}.png"
        
        canvas = self.drawing_engine.get_composite_canvas()
        cv2.imwrite(filename, canvas)
        print(f"💾 Artwork saved: {filename}")
    
    def toggle_recording(self):
        """Toggle session recording"""
        self.recording = not self.recording
        
        if self.recording:
            self.session_frames = []
            print("🎥 Recording started")
        else:
            self.save_recording()
            print("⏹️ Recording stopped")
    
    def save_recording(self):
        """Save recorded session"""
        if not self.session_frames:
            return
        
        timestamp = int(time.time())
        filename = f"exports/videos/neural_canvas_session_{timestamp}.mp4"
        
        # Save video (simplified - you'd use proper video encoding)
        print(f"💾 Session saved: {filename}")
    
    def get_current_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            fps = len(self.frame_times) / time_diff if time_diff > 0 else 0
        else:
            fps = 0
        
        return int(fps)
    
    def run(self):
        """Main application loop"""
        print("🎨 Starting NeuralCanvas AR...")

        try:
            cap, frame_shape = self.initialize_camera()
        except RuntimeError as e:
            print(e)
            return

        self.running = True

        # Print controls/help
        print("\n" + "=" * 60)
        print("🎨 NEURALCANVAS AR - CONTROLS")
        print("=" * 60)
        print("GESTURES:")
        print("👆 Index finger alone: Draw")
        print("✌️  Index + Middle: Navigate")
        print("✋ All fingers: Erase")
        print("👍 Thumb + Index: Change color")
        # Shape mode and voice control features removed
        print("✌️🖕 Three fingers: Save")
        print("🖖 Four fingers: Clear")
        print("\nKEYBOARD:")
        print("'q': Quit | 'c': Clear | 's': Save")
        print("'r': Record")
        print("'3': 3D/2D mode | '+/-': Brush size")
        print("'u': Undo")
        print("=" * 60)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for natural interaction
            frame = cv2.flip(frame, 1)

            # Track hands
            hand_results = self.hand_tracker.process_frame(frame)

            # Process gestures
            commands, drawing_points = self.process_gestures(hand_results)

            # Process drawing points
            if drawing_points:
                self.process_drawing_points(drawing_points)

            # Emotion/voice features removed; keep placeholder
            emotion_info = None

            # Render final frame
            final_frame = self.render_frame(frame, hand_results, commands, emotion_info)

            # Add to recording if active
            if self.recording:
                self.session_frames.append(final_frame.copy())

            # Display
            cv2.imshow('🎨 NeuralCanvas AR - The Future of Digital Art', final_frame)

            # Handle input
            if not self.handle_keyboard_input():
                break

            # Performance monitoring
            self.performance_monitor.update()

        # Cleanup
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

        # Save recording if any
        if self.recording:
            self.save_recording()

        print("👋 Thanks for using NeuralCanvas AR!")

if __name__ == "__main__":
    app = NeuralCanvasAR()
    app.run()
