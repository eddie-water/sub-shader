"""
Utility modules for SubShader.

This package contains various utility modules for environment setup,
performance monitoring, logging configuration, and other helper functionality.
"""

from .logging import logger_init, get_logger, set_log_level, get_module_logger
from .os_env_setup import env_init
from .loop_timer import LoopTimer
from .gpu import gpu_available
from .timing import timed

__all__ = [
    'env_init',
    'LoopTimer',
    'logger_init',
    'get_logger',
    'set_log_level',
    'get_module_logger',
    'gpu_available',
    'timed',
]