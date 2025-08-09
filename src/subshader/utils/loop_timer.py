import time
from typing import List
from subshader.utils.logging import get_logger

log = get_logger(__name__)


class LoopTimer:
    """
    Utility class for timing main loop iterations and calculating iterations per second.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize loop timer.
        
        Args:
            update_interval (float): How often to calculate and report iterations per second (in seconds)
        """
        self.update_interval = update_interval
        self.loop_timer = time.perf_counter()
        self.iteration_times: List[float] = []
    
    def start_loop(self) -> float:
        """
        Start timing a loop.
        
        Returns:
            float: Current timestamp for loop start
        """
        return time.perf_counter()
    
    def end_loop_and_report(self, loop_start: float) -> float:
        """
        End timing a loop, record the duration, and report loops per second if needed.
        
        Args:
            loop_start (float): Timestamp from start_loop()
            
        Returns:
            float: Loop duration in seconds
        """
        loop_end = time.perf_counter()
        loop_duration = loop_end - loop_start
        self.iteration_times.append(loop_duration)
        
        # Check if it's time to report loops per second
        if time.time() - self.loop_timer > self.update_interval and len(self.iteration_times) > 0:
            loops_per_sec = self.get_loops_per_second()
            log.info(f"Loops per second: {loops_per_sec:.2f}")
            self.iteration_times.clear()
            self.loop_timer = time.time()
        
        return loop_duration
    
    def get_loops_per_second(self) -> float:
        """
        Calculate current loops per second based on recorded loop times.
        
        Returns:
            float: Current loops per second
        """
        if not self.iteration_times:
            return 0.0
        
        avg_loop_time = sum(self.iteration_times) / len(self.iteration_times)
        return 1.0 / avg_loop_time if avg_loop_time > 0 else 0.0 