"""Custom exceptions and exception handling for SubShader."""

from typing import Type
from subshader.utils.logging import get_logger

log = get_logger(__name__)


class SubShaderException(Exception):
    """Base exception for SubShader application."""
    log_level = "info"
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class EndOfAudioException(SubShaderException):
    """Raised when audio file reaches end."""
    log_level = "info"


class WindowCloseException(SubShaderException):
    """Raised when visualization window is closed."""
    log_level = "info"


class AudioFileNotFoundError(SubShaderException):
    """Raised when audio file cannot be found."""
    log_level = "error"

# Exception tuple for catching
GRACEFUL_EXCEPTIONS = (
    SubShaderException,
    KeyboardInterrupt,
)


class ExceptionReporter:
    """Handles logging of SubShader exceptions."""
    
    @staticmethod
    def report(e: Exception) -> None:
        """Log exception with appropriate level and message."""
        if isinstance(e, KeyboardInterrupt):
            log.warning(f"Keyboard interrupt received {type(e).__name__}: {e}")
        elif isinstance(e, SubShaderException):
            getattr(log, e.log_level)(e.message)
        else:
            log.error(f"Unexpected error: {type(e).__name__}: {e}")

# Singleton instance
reporter = ExceptionReporter()