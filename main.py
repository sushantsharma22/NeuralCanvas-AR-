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
from modules.face_analyzer import EmotionColorMapper
from modules.voice_controller import VoiceController
from utils.performance_monitor import PerformanceMonitor
from ui.interface_manager import InterfaceManager

class NeuralCanvasAR:
    def __init__(self):
        print("🚀 Initializing NeuralCanvas AR - The Future of Digital Art!")
        
        # Core components
        self.gesture_engine = AdvancedGestureEngine()
        self.hand_tracker = AdvancedHandTracker(max_hands=Config.MAX_HANDS)
        self.emotion_mapper = EmotionColorMapper()
        self.performance_monitor = PerformanceMonitor()
        self.voice_controller = VoiceController()
        
        # Drawing system
        self.drawing_engine = None
        self.ui_manager = None
        
        # Application state
        self.running = False
        self.current_mode = "2D"
        self.voice_enabled = False
        self.emotion_coloring = True
        self.auto_save = True
        
        # Drawing state tracking
        self.drawing_active = False
        self.last_gesture = None
        self.last_drawing_point = None
        self.gesture_history = deque(maxlen=5)  # Track recent gestures for stability
        self.drawing_timeout = 0  # Frame counter for drawing timeout
        
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
            if self.drawing_active:
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
            
            # Recognize gesture
            gesture, confidence = self.gesture_engine.recognize_gesture(landmarks.landmark)
            
            # Debug output for gesture detection (reduce spam)
            if gesture != "NONE" and gesture != "COLOR_CHANGE":
                print(f"🎯 Detected: {gesture} (confidence: {confidence:.2f})")
            
            # CRITICAL: Handle IDLE state (closed hand) - STOP DRAWING IMMEDIATELY
            if gesture == "IDLE":
                if self.drawing_active:
                    print("🛑 Hand closed - ending drawing stroke")
                    self.drawing_engine.end_stroke()
                    self.drawing_active = False
                continue  # Skip further processing for this hand
            
            # CRITICAL: If not drawing gesture and not high confidence, DON'T continue drawing
            if gesture not in ["DRAW", "ERASE"] and self.drawing_active:
                # Only maintain drawing if we have very recent DRAW history AND decent confidence
                recent_draws = sum(1 for g in list(self.gesture_history)[-3:] if g == "DRAW")
                if recent_draws < 2 or confidence < 0.4:
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
        h, w = self.drawing_engine.height, self.drawing_engine.width
        
        if gesture == "COLOR_CHANGE":
            # Cycle through colors
            colors = list(Config.COLORS.values())
            current_idx = colors.index(self.drawing_engine.current_color) if self.drawing_engine.current_color in colors else 0
            next_color = colors[(current_idx + 1) % len(colors)]
            self.drawing_engine.change_color(next_color)
            return f"Changed color to {list(Config.COLORS.keys())[colors.index(next_color)]}"
        
        elif gesture == "VOICE_ACTIVATE":
            self.voice_enabled = not self.voice_enabled
            if self.voice_enabled:
                self.voice_controller.start_listening()
            else:
                self.voice_controller.stop_listening()
            return f"Voice control {'enabled' if self.voice_enabled else 'disabled'}"
        
        elif gesture == "SAVE":
            self.save_artwork()
            return "Artwork saved"
        
        elif gesture == "CLEAR":
            self.drawing_engine.clear_canvas()
            return "Canvas cleared"
        
        elif gesture == "SHAPE_MODE":
            # Toggle shape recognition
            self.drawing_engine.shape_recognition_enabled = not self.drawing_engine.shape_recognition_enabled
            return f"Shape recognition {'enabled' if self.drawing_engine.shape_recognition_enabled else 'disabled'}"
        
        return None
    
    def process_voice_command(self, command):
        """Process voice commands and return status message"""
        if command == "DRAW":
            # Force drawing mode for next gesture
            return "Voice: Drawing mode activated"
        
        elif command == "ERASE":
            self.drawing_engine.clear_canvas()
            return "Voice: Canvas cleared"
        
        elif command == "CLEAR":
            self.drawing_engine.clear_canvas()
            return "Voice: Canvas cleared"
        
        elif command == "UNDO":
            # Add undo functionality if available
            return "Voice: Undo (not implemented yet)"
        
        elif command.startswith("COLOR_"):
            color_name = command.split("_")[1].lower()
            color_map = {
                'red': (0, 0, 255),
                'blue': (255, 0, 0),
                'green': (0, 255, 0),
                'black': (0, 0, 0),
                'white': (255, 255, 255),
                'yellow': (0, 255, 255)
            }
            if color_name in color_map:
                self.drawing_engine.change_color(color_map[color_name])
                return f"Voice: Changed color to {color_name}"
        
        elif command == "SAVE":
            self.save_artwork()
            return "Voice: Artwork saved"
        
        elif command == "BRUSH_BIGGER":
            current_size = getattr(self.drawing_engine, 'brush_size', 4)
            new_size = min(current_size + 2, 20)
            self.drawing_engine.brush_size = new_size
            return f"Voice: Brush size increased to {new_size}"
        
        elif command == "BRUSH_SMALLER":
            current_size = getattr(self.drawing_engine, 'brush_size', 4)
            new_size = max(current_size - 2, 1)
            self.drawing_engine.brush_size = new_size
            return f"Voice: Brush size decreased to {new_size}"
        
        elif command == "STOP" or command == "QUIT":
            self.running = False
            return "Voice: Stopping application"
        
        return f"Voice: Unknown command '{command}'"
    
    def extract_drawing_point(self, landmarks, gesture, hand_id):
        """Extract drawing point from hand landmarks"""
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
        if not drawing_points and self.drawing_active and self.drawing_timeout > 3:
            print("🎨 Ending drawing stroke (gesture timeout)")
            self.drawing_engine.end_stroke()
            self.drawing_active = False
            self.last_gesture = None
    
    def update_emotion_coloring(self, frame):
        """Update drawing color based on emotion"""
        if self.emotion_coloring:
            emotion, color = self.emotion_mapper.analyze_emotion(frame)
            
            # Gradually transition to emotion color
            current_color = self.drawing_engine.current_color
            blend_factor = 0.1
            
            new_color = tuple(
                int(c * (1 - blend_factor) + e * blend_factor)
                for c, e in zip(current_color, color)
            )
            
            self.drawing_engine.change_color(new_color)
            return emotion, color
        
        return None, None
    
    def render_frame(self, frame, hand_results, commands, emotion_info):
        """Render final frame with all overlays"""
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
        elif key == ord('c'):
            self.drawing_engine.clear_canvas()
        elif key == ord('s'):
            self.save_artwork()
        elif key == ord('r'):
            self.toggle_recording()
        elif key == ord('e'):
            self.emotion_coloring = not self.emotion_coloring
        elif key == ord('3'):
            self.current_mode = "3D" if self.current_mode == "2D" else "2D"
        elif key == ord('=') or key == ord('+'):
            new_size = min(self.drawing_engine.brush_size + 2, Config.MAX_BRUSH_SIZE)
            self.drawing_engine.change_brush_size(new_size)
        elif key == ord('-'):
            new_size = max(self.drawing_engine.brush_size - 2, Config.MIN_BRUSH_SIZE)
            self.drawing_engine.change_brush_size(new_size)
        elif key == ord('u'):
            self.drawing_engine.undo()
        
        return True
    
    def save_artwork(self):
        """Save current artwork"""
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
        
        print("\n" + "="*60)
        print("🎨 NEURALCANVAS AR - CONTROLS")
        print("="*60)
        print("GESTURES:")
        print("👆 Index finger alone: Draw")
        print("✌️  Index + Middle: Navigate")
        print("✋ All fingers: Erase")
        print("👍 Thumb + Index: Change color")
        print("🤟 Index + Pinky: Toggle shape mode")
        print("🤙 Thumb + Pinky: Voice control")
        print("✌️🖕 Three fingers: Save")
        print("🖖 Four fingers: Clear")
        print("\nKEYBOARD:")
        print("'q': Quit | 'c': Clear | 's': Save")
        print("'r': Record | 'e': Emotion coloring")
        print("'3': 3D/2D mode | '+/-': Brush size")
        print("'u': Undo")
        print("="*60)
        
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
            
            # Process voice commands if enabled
            if self.voice_enabled:
                voice_command = self.voice_controller.get_command()
                if voice_command:
                    voice_result = self.process_voice_command(voice_command)
                    if voice_result:
                        commands.append(voice_result)
            
            # Process drawing
            if drawing_points:
                self.process_drawing_points(drawing_points)
            
            # Update emotion coloring
            emotion_info = self.update_emotion_coloring(frame)
            
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
        cap.release()
        cv2.destroyAllWindows()
        
        # Stop voice controller
        if self.voice_enabled:
            self.voice_controller.stop_listening()
        
        if self.recording:
            self.save_recording()
        
        print("👋 Thanks for using NeuralCanvas AR!")

if __name__ == "__main__":
    app = NeuralCanvasAR()
    app.run()
