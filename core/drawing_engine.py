"""
Advanced 3D drawing engine with AI-assisted features
"""
import numpy as np
import cv2
from collections import deque
import math

class NeuralDrawingEngine:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        
        # Multi-layer canvas system
        self.canvas_layers = [np.zeros((self.height, self.width, 3), dtype=np.uint8) for _ in range(10)]
        self.current_layer = 0
        
        # Drawing state
        self.current_color = (255, 255, 255)  # White
        self.brush_size = 4  # Reduced default brush size
        self.z_position = 5
        self.is_drawing = False
        self.last_point = None
        
        # Smoothing and buffering
        self.smoothing_enabled = True
        self.point_buffer = deque(maxlen=5)  # Reduced buffer for more responsive drawing
        self.stroke_history = []
        
        # Advanced drawing features
        self.pressure_sensitivity = True
        self.depth_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Performance tracking
        self.undo_stack = []
        
        # AI-assisted drawing
        self.shape_recognition_enabled = True
        self.auto_complete_enabled = True
        self.stroke_prediction_enabled = True

        # Effects and filters (disable heavy effects for better performance)
        self.glow_effect = False  # Disable glow for cleaner drawing
        self.particle_effects = False  # Disable particles for performance
        self.trail_effects = True

        # Debugging flag (turn on to visualize buffer points)
        self.debug_draw = False
        
    def draw_point(self, x, y, pressure=1.0, z=None):
        """Draw a point with advanced smoothing and line continuation"""
        if z is None:
            z = self.z_position
            
        # Clamp coordinates
        x, y = max(0, min(x, self.width-1)), max(0, min(y, self.height-1))
        
        # Calculate dynamic brush size based on pressure (cap the maximum)
        base_size = min(self.brush_size, 20)  # Cap brush size
        dynamic_size = max(1, int(base_size * pressure)) if self.pressure_sensitivity else base_size
        
        # Apply progressive smoothing to reduce jitter
        if self.last_point is not None:
            last_x, last_y = self.last_point[0], self.last_point[1]
            
            # Calculate velocity for adaptive smoothing
            velocity = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            
            # Apply stronger smoothing for high velocity (fast movements)
            if velocity > 15:  # Fast movement threshold
                smoothing_factor = 0.4  # More aggressive smoothing
                x = int(last_x + (x - last_x) * smoothing_factor)
                y = int(last_y + (y - last_y) * smoothing_factor)
            elif velocity > 8:  # Medium movement
                smoothing_factor = 0.7
                x = int(last_x + (x - last_x) * smoothing_factor)
                y = int(last_y + (y - last_y) * smoothing_factor)
            # For slow movements (velocity <= 8), use original coordinates
        
        current_point = (x, y, dynamic_size, pressure, z)
        
        # If we have a last point and are drawing, always connect with a line
        if self.last_point is not None and self.is_drawing:
            self._draw_smooth_connected_line(self.last_point, current_point)
        else:
            # First point of a stroke - draw the point with extra smoothness
            self._draw_raw_point(x, y, dynamic_size, z)
            self.is_drawing = True
        
        # Update last point for next connection
        self.last_point = current_point
        
        # Add to stroke history
        self.stroke_history.append({
            'x': x, 'y': y, 'size': dynamic_size, 'color': self.current_color,
            'z': z, 'timestamp': cv2.getTickCount()
        })

    def start_stroke(self, x, y, pressure=1.0, z=None):
        """Start a new drawing stroke"""
        self.is_drawing = True
        self.last_point = None
        self.point_buffer.clear()
        self.draw_point(x, y, pressure, z)

    def end_stroke(self):
        """End the current drawing stroke"""
        self.is_drawing = False
        self.last_point = None
        self.point_buffer.clear()

    def _draw_smooth_connected_line(self, start_point, end_point):
        """Draw a smooth line between two points using optimized OpenCV line drawing"""
        x1, y1, size1, pressure1, z1 = start_point
        x2, y2, size2, pressure2, z2 = end_point
        
        # Calculate distance between points
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Always draw connection lines, even for very close points
        if distance < 0.5:
            self._draw_raw_point(int(x2), int(y2), size2, z2)
            return
        
        # Use average size for the line
        avg_size = max(1, int((size1 + size2) / 2))
        avg_z = (z1 + z2) / 2
        
        # Get the canvas to draw on
        layer_index = int(avg_z) % len(self.canvas_layers)
        canvas = self.canvas_layers[layer_index]
        
        # For smooth drawing, use multiple line passes
        if avg_size <= 2:
            # Thin lines - use anti-aliased line
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), self.current_color, avg_size, cv2.LINE_AA)
        else:
            # Thicker lines - draw filled circle path for ultra-smooth appearance
            num_steps = max(2, int(distance))
            for i in range(num_steps + 1):
                t = i / num_steps
                x = int(x1 + (x2 - x1) * t)
                y = int(y1 + (y2 - y1) * t)
                size = int(size1 + (size2 - size1) * t)
                
                # Ensure coordinates are within bounds
                x = max(size, min(x, self.width - size - 1))
                y = max(size, min(y, self.height - size - 1))
                
                # Draw small circles along the path for ultra-smooth appearance
                cv2.circle(canvas, (x, y), size, self.current_color, -1, cv2.LINE_AA)
        
        # Update depth map along the line
        cv2.line(self.depth_map, (int(x1), int(y1)), (int(x2), int(y2)), float(avg_z), avg_size)
    
    def _draw_smooth_line(self):
        """Draw smooth line using Catmull-Rom spline interpolation"""
        if len(self.point_buffer) < 3:
            return

        points = list(self.point_buffer)

        # Decide on 4 control points for Catmull-Rom when available
        if len(points) >= 4:
            p0, p1, p2, p3 = points[-4], points[-3], points[-2], points[-1]
        else:
            # Duplicate ends to form a 4-tuple
            p0, p1, p2 = points[-3], points[-2], points[-1]
            p3 = p2

        # Draw the middle endpoint to avoid gaps
        end_x, end_y, end_size, _, end_z = p2
        self._draw_raw_point(int(end_x), int(end_y), max(1, int(end_size)), float(end_z))

        # Catmull-Rom interpolation using scalar coordinates between p1 and p2
        num_segments = 8
        for i in range(1, num_segments + 1):
            t = i / (num_segments + 1)

            x = self._catmull_rom_interpolate(p0[0], p1[0], p2[0], p3[0], t)
            y = self._catmull_rom_interpolate(p0[1], p1[1], p2[1], p3[1], t)

            size = max(1, int(self._linear_interpolate(p1[2], p2[2], t)))
            z = float(self._linear_interpolate(p1[4], p2[4], t))

            self._draw_raw_point(int(x), int(y), size, z)
    
    def _catmull_rom_interpolate(self, p0, p1, p2, p3, t):
        """Catmull-Rom spline interpolation"""
        return 0.5 * (
            2 * p1 +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
        )
    
    def _linear_interpolate(self, a, b, t):
        """Linear interpolation"""
        return a + (b - a) * t
    
    def _draw_raw_point(self, x, y, size, z):
        """Draw raw point with smooth edges"""
        layer_index = int(z) % len(self.canvas_layers)
        canvas = self.canvas_layers[layer_index]
        
        # Ensure coordinates are within bounds
        x = max(size, min(x, self.width - size - 1))
        y = max(size, min(y, self.height - size - 1))
        
        # Main brush stroke with anti-aliasing for smooth edges
        if self.glow_effect:
            self._draw_glow_circle(canvas, x, y, size)
        else:
            # Use anti-aliased circle for smoother appearance
            cv2.circle(canvas, (x, y), size, self.current_color, -1, cv2.LINE_AA)
            
            # Add subtle feathering for very smooth edges
            if size > 2:
                # Draw a slightly larger, more transparent circle for feathering
                feather_color = tuple(int(c * 0.3) for c in self.current_color)
                cv2.circle(canvas, (x, y), size + 1, feather_color, 1, cv2.LINE_AA)

        # Update depth map
        cv2.circle(self.depth_map, (x, y), size, float(z), -1)

        # Particle effects (if enabled)
        if self.particle_effects:
            self._add_particles(canvas, x, y, size)
        
        # Add to stroke history
        self.stroke_history.append({
            'x': x, 'y': y, 'size': size, 'color': self.current_color,
            'z': z, 'timestamp': cv2.getTickCount()
        })

    def erase_at_point(self, x, y, thickness=20):
        """Simple eraser that clears circles on all layers around a point"""
        for layer in self.canvas_layers:
            cv2.circle(layer, (x, y), thickness, (0, 0, 0), -1)
        # Update depth map too
        cv2.circle(self.depth_map, (x, y), thickness, 0.0, -1)
    
    def _draw_glow_circle(self, canvas, x, y, size):
        """Draw circle with glow effect"""
        # Create multiple layers for glow
        glow_layers = 5
        for i in range(glow_layers, 0, -1):
            glow_size = size + i * 3
            alpha = 0.3 / i
            
            # Create temporary overlay
            overlay = canvas.copy()
            cv2.circle(overlay, (x, y), glow_size, self.current_color, -1)
            
            # Blend with canvas
            cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0, canvas)
        
        # Draw main circle
        cv2.circle(canvas, (x, y), size, self.current_color, -1)
    
    def _add_particles(self, canvas, x, y, size):
        """Add particle effects around brush"""
        num_particles = min(size // 3, 10)
        
        for _ in range(num_particles):
            # Random particle position around brush
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(size, size * 2)
            
            px = int(x + distance * np.cos(angle))
            py = int(y + distance * np.sin(angle))
            
            # Check bounds
            if 0 <= px < self.width and 0 <= py < self.height:
                particle_size = np.random.randint(1, 3)
                particle_color = tuple(int(c * np.random.uniform(0.5, 1.0)) for c in self.current_color)
                cv2.circle(canvas, (px, py), particle_size, particle_color, -1)
    
    def recognize_shape(self, points):
        """Recognize and auto-complete geometric shapes"""
        if not self.shape_recognition_enabled or len(points) < 10:
            return None
        
        # Convert points to numpy array
        points_array = np.array([(p['x'], p['y']) for p in points])
        
        # Circle detection
        if self._is_circle(points_array):
            return self._complete_circle(points_array)
        
        # Rectangle detection
        elif self._is_rectangle(points_array):
            return self._complete_rectangle(points_array)
        
        # Line detection
        elif self._is_line(points_array):
            return self._complete_line(points_array)
        
        return None
    
    def _is_circle(self, points):
        """Detect if points form a circle"""
        if len(points) < 8:
            return False
        
        # Calculate center and radius
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        
        # Check if distances are relatively consistent
        std_dev = np.std(distances)
        mean_distance = np.mean(distances)
        
        return std_dev < mean_distance * 0.3
    
    def _complete_circle(self, points):
        """Complete circle shape"""
        center = np.mean(points, axis=0)
        radius = int(np.mean(np.linalg.norm(points - center, axis=1)))
        
        return {
            'type': 'circle',
            'center': center.astype(int),
            'radius': radius
        }
    
    def _is_rectangle(self, points):
        """Detect if points form a rectangle"""
        if len(points) < 8:
            return False
        
        # Find corners using convex hull
        hull = cv2.convexHull(points.astype(np.float32))
        
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        return len(approx) == 4
    
    def _complete_rectangle(self, points):
        """Complete rectangle shape"""
        hull = cv2.convexHull(points.astype(np.float32))
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        return {
            'type': 'rectangle',
            'corners': approx.reshape(-1, 2).astype(int)
        }
    
    def _is_line(self, points):
        """Detect if points form a line"""
        if len(points) < 5:
            return False
        
        # Fit line using least squares
        A = np.vstack([points[:, 0], np.ones(len(points))]).T
        m, b = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
        
        # Calculate distances from line
        line_points = m * points[:, 0] + b
        distances = np.abs(points[:, 1] - line_points)
        
        # Check if points are close to the line
        return np.mean(distances) < 10
    
    def _complete_line(self, points):
        """Complete line shape"""
        start_point = points
        end_point = points[-1]
        
        return {
            'type': 'line',
            'start': start_point.astype(int),
            'end': end_point.astype(int)
        }
    
    def draw_shape(self, shape_info):
        """Draw completed shape"""
        canvas = self.canvas_layers[self.current_layer]
        
        if shape_info['type'] == 'circle':
            cv2.circle(canvas, tuple(shape_info['center']), shape_info['radius'], 
                      self.current_color, self.brush_size)
        
        elif shape_info['type'] == 'rectangle':
            corners = shape_info['corners']
            cv2.polylines(canvas, [corners], True, self.current_color, self.brush_size)
        
        elif shape_info['type'] == 'line':
            cv2.line(canvas, tuple(shape_info['start']), tuple(shape_info['end']), 
                    self.current_color, self.brush_size)
    
    def get_composite_canvas(self):
        """Combine all layers into final image"""
        result = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for i, layer in enumerate(self.canvas_layers):
            # Apply depth-based alpha blending
            alpha = 1.0 - (i * 0.1)  # Deeper layers are more transparent
            result = cv2.addWeighted(result, 1, layer, alpha, 0)
        
        return result
    
    def save_state(self):
        """Save current canvas state for undo"""
        state = [layer.copy() for layer in self.canvas_layers]
        self.undo_stack.append(state)
        
        if len(self.undo_stack) > 50:
            self.undo_stack.popleft()
    
    def undo(self):
        """Undo last action"""
        if self.undo_stack:
            self.canvas_layers = self.undo_stack.pop()
    
    def clear_canvas(self):
        """Clear all drawing layers"""
        self.save_state()
        for layer in self.canvas_layers:
            layer.fill(0)
        self.depth_map.fill(0)
        self.stroke_history.clear()
    
    def change_color(self, color):
        """Change drawing color"""
        self.current_color = color
    
    def change_brush_size(self, size):
        """Change brush size"""
        self.brush_size = max(2, min(size, 100))
    
    def change_layer(self, layer):
        """Change active drawing layer"""
        self.current_layer = max(0, min(layer, len(self.canvas_layers) - 1))