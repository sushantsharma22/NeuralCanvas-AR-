"""
Performance monitoring and optimization
"""
import time
import psutil
import cv2
from collections import deque


class PerformanceMonitor:
    def __init__(self):
        self.frame_times = deque(maxlen=60)
        self.cpu_usage = deque(maxlen=30)
        self.memory_usage = deque(maxlen=30)
        self.last_update = time.time()

    def update(self):
        """Update performance metrics"""
        current_time = time.time()
        self.frame_times.append(current_time)

        # Update system metrics every second
        if current_time - self.last_update > 1.0:
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().percent)
            self.last_update = current_time

    def get_fps(self):
        """Calculate current FPS"""
        if len(self.frame_times) < 2:
            return 0

        time_span = self.frame_times[-1] - self.frame_times[0]
        return len(self.frame_times) / time_span if time_span > 0 else 0

    def draw_stats(self, frame):
        """Draw performance statistics"""
        fps = int(self.get_fps())
        cpu = int(sum(self.cpu_usage) / len(self.cpu_usage)) if self.cpu_usage else 0
        memory = int(sum(self.memory_usage) / len(self.memory_usage)) if self.memory_usage else 0
        # Performance box
        try:
            h, w = frame.shape[:2]
        except Exception:
            return frame

        box_height = 120
        box_width = 200
        y_start = h - box_height - 10
        x_start = w - box_width - 10

        # Background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + box_width, y_start + box_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(
            frame,
            (x_start, y_start),
            (x_start + box_width, y_start + box_height),
            (255, 255, 255),
            2,
        )

        # Text
        cv2.putText(
            frame,
            "PERFORMANCE",
            (x_start + 10, y_start + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # FPS with color coding
        fps_color = (0, 255, 0) if fps > 30 else (0, 255, 255) if fps > 15 else (0, 0, 255)
        cv2.putText(
            frame,
            f"FPS: {fps}",
            (x_start + 10, y_start + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            fps_color,
            1,
        )

        # CPU
        cpu_color = (0, 255, 0) if cpu < 50 else (0, 255, 255) if cpu < 80 else (0, 0, 255)
        cv2.putText(
            frame,
            f"CPU: {cpu}%",
            (x_start + 10, y_start + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            cpu_color,
            1,
        )

        # Memory
        mem_color = (0, 255, 0) if memory < 60 else (0, 255, 255) if memory < 85 else (0, 0, 255)
        cv2.putText(
            frame,
            f"MEM: {memory}%",
            (x_start + 10, y_start + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            mem_color,
            1,
        )

        return frame
