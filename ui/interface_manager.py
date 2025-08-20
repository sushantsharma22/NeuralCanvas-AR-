"""
Advanced user interface management
"""
import cv2
import numpy as np
from config.settings import Config

class InterfaceManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.setup_ui_elements()
        
    def setup_ui_elements(self):
        """Setup UI element positions"""
        # Color palette
        self.color_palette_rect = (50, 10, 600, 60)
        
        # Tool panel
        self.tool_panel_rect = (self.width - 200, 10, 180, 400)
        
        # Status bar
        self.status_bar_rect = (10, self.height - 100, self.width - 20, 80)
        
    def draw_interface(self, frame, app_state):
        """Draw complete user interface"""
        self.draw_color_palette(frame, app_state['current_color'])
        self.draw_tool_panel(frame, app_state)
        self.draw_status_bar(frame, app_state)
        self.draw_command_feedback(frame, app_state.get('commands', []))
        
    def draw_color_palette(self, frame, current_color):
        """Draw advanced color palette"""
        x, y, w, h = self.color_palette_rect
        
        # Background with glass effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x-10, y-10), (x+w+10, y+h+10), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border with glow effect
        cv2.rectangle(frame, (x-12, y-12), (x+w+12, y+h+12), (100, 100, 100), 2)
        cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (200, 200, 200), 1)
        
        # Color buttons
        colors = list(Config.COLORS.items())
        button_width = w // len(colors)
        
        for i, (name, color) in enumerate(colors):
            bx = x + i * button_width
            by = y
            bw = button_width - 5
            bh = h - 10
            
            # Button background with gradient effect
            self.draw_gradient_button(frame, bx, by, bw, bh, color)
            
            # Highlight current color
            if np.array_equal(color, current_color):
                cv2.rectangle(frame, (bx-5, by-5), (bx+bw+5, by+bh+5), (255, 255, 255), 3)
                cv2.rectangle(frame, (bx-3, by-3), (bx+bw+3, by+bh+3), color, 2)
            
            # Color name
            text_size = cv2.getTextSize(name.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            text_w = text_size[0]
            text_x = int(bx + (bw - text_w) // 2)
            text_y = int(by + bh + 15)

            cv2.putText(frame, name.upper(), (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def draw_gradient_button(self, frame, x, y, w, h, color):
        """Draw button with gradient effect"""
        # Create gradient
        for i in range(h):
            alpha = 0.7 + 0.3 * (i / h)  # Gradient from 0.7 to 1.0
            row_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, (x, y + i), (x + w, y + i), row_color, 1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
    
    def draw_tool_panel(self, frame, app_state):
        """Draw advanced tool panel"""
        x, y, w, h = self.tool_panel_rect
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Panel border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "TOOLS", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Brush size indicator
        brush_size = app_state['brush_size']
        cv2.putText(frame, f"Brush: {brush_size}", (x + 10, y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Brush preview
        brush_x, brush_y = x + w - 40, y + 45
        cv2.circle(frame, (brush_x, brush_y), brush_size, app_state['current_color'], -1)
        cv2.circle(frame, (brush_x, brush_y), brush_size + 2, (255, 255, 255), 1)
        
        # Current layer
        cv2.putText(frame, f"Layer: {app_state['current_layer'] + 1}", (x + 10, y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mode indicator
        mode = app_state['mode']
        mode_color = (0, 255, 255) if mode == "3D" else (255, 255, 255)
        cv2.putText(frame, f"Mode: {mode}", (x + 10, y + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        # FPS indicator
        fps = app_state['fps']
        fps_color = (0, 255, 0) if fps > 30 else (255, 255, 0) if fps > 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps}", (x + 10, y + 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        
        # Gesture instructions
        instructions = [
            "GESTURES:",
            "👆 Draw", "✌️ Navigate", "✋ Erase",
            "👍 Color", "🤟 Shape", "🤙 Voice"
        ]
        
        for i, instruction in enumerate(instructions):
            color = (255, 255, 0) if i == 0 else (200, 200, 200)
            font_scale = 0.4 if i == 0 else 0.3
            cv2.putText(frame, instruction, (x + 10, y + 190 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    
    def draw_status_bar(self, frame, app_state):
        """Draw status bar with current information"""
        x, y, w, h = self.status_bar_rect
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
        
        # Emotion info
        emotion_info = app_state.get('emotion_info')
        if emotion_info:
            emotion, color = emotion_info
            cv2.putText(frame, f"Emotion: {emotion.capitalize()}", (x + 10, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Emotion color indicator
            cv2.rectangle(frame, (x + 200, y + 10), (x + 240, y + 30), color, -1)
            cv2.rectangle(frame, (x + 200, y + 10), (x + 240, y + 30), (255, 255, 255), 1)
        
        # Keyboard shortcuts
        shortcuts = "Q:Quit | C:Clear | S:Save | R:Record | E:Emotion | 3:3D/2D | +/-:Brush | U:Undo"
        cv2.putText(frame, shortcuts, (x + 10, y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_command_feedback(self, frame, commands):
        """Draw command execution feedback"""
        if not commands:
            return
        
        # Show latest commands
        y_offset = 100
        for i, command in enumerate(commands[-3:]):  # Show last 3 commands
            alpha = 1.0 - (i * 0.3)  # Fade older commands
            
            # Background
            text_size = cv2.getTextSize(command, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_w = text_size[0]
            x = int((frame.shape[1] - text_w) // 2)
            y = y_offset + i * 40

            overlay = frame.copy()
            cv2.rectangle(overlay, (x - 10, y - 25), (x + text_w + 10, y + 5),
                         (0, 200, 0), -1)
            cv2.addWeighted(overlay, alpha * 0.8, frame, 1 - alpha * 0.8, 0, frame)
            
            # Text
            text_color = tuple(int(255 * alpha) for _ in range(3))
            cv2.putText(frame, command, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
