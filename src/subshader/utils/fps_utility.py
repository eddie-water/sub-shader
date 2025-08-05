import time
from typing import List


class FpsUtility:
    """
    Utility class for handling FPS timing and calculation.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize FPS utility.
        
        Args:
            update_interval (float): How often to calculate and report FPS (in seconds)
        """
        self.update_interval = update_interval
        self.fps_timer = time.perf_counter()
        self.frame_times: List[float] = []
    
    def start_frame(self) -> float:
        """
        Start timing a frame.
        
        Returns:
            float: Current timestamp for frame start
        """
        return time.perf_counter()
    
    def end_frame_and_report(self, frame_start: float) -> float:
        """
        End timing a frame, record the duration, and report FPS if needed.
        
        Args:
            frame_start (float): Timestamp from start_frame()
            
        Returns:
            float: Frame duration in seconds
        """
        frame_end = time.perf_counter()
        frame_duration = frame_end - frame_start
        self.frame_times.append(frame_duration)
        
        # Check if it's time to report FPS
        if time.time() - self.fps_timer > self.update_interval and len(self.frame_times) > 0:
            fps = self.get_fps()
            print(f"FPS: {fps:.2f}")
            self.frame_times.clear()
            self.fps_timer = time.time()
        
        return frame_duration
    
    def get_fps(self) -> float:
        """
        Calculate current FPS based on recorded frame times.
        
        Returns:
            float: Current FPS
        """
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0 