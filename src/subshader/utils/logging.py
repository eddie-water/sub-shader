"""
Centralized logging configuration for SubShader.

This module provides a unified logging setup that can be accessed
from anywhere in the application.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def logger_init(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True
) -> None:
    """
    Set up centralized logging for the entire application.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (Optional[str]): Path to log file. If None, uses default.
        console_output (bool): Whether to output logs to console
        file_output (bool): Whether to output logs to file
    """
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "subshader.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root log
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_output:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log the setup
    log = logging.getLogger(__name__)
    log.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a log instance for a specific module.
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured log instance
    """
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """
    Change the logging level for all handlers.
    
    Args:
        level (str): New logging level
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    
    log = logging.getLogger(__name__)
    log.info(f"Log level changed to: {level}")


# Convenience function for quick log access
def get_module_logger() -> logging.Logger:
    """
    Get a log for the calling module.
    
    Returns:
        logging.Logger: Logger for the calling module
    """
    import inspect
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals.get('__name__', 'unknown')
    return get_logger(module_name) 