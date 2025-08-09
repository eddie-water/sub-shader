# Logging Guide for SubShader

This guide explains the centralized logging system for SubShader.

## Overview

The logging system is centralized in `src/subshader/utils/logging.py` and provides unified logging across the application.

## Quick Start

### Initialization
```python
from subshader.utils.logging import logger_init, get_logger

# Initialize logging
logger_init(log_level="INFO", console_output=True, file_output=True)
log = get_logger(__name__)
```

### Usage in Modules
```python
from subshader.utils.logging import get_logger

log = get_logger(__name__)

log.info("Application started")
log.debug("Processing frame 42")
log.warning("End of audio file reached")
log.error("GPU memory allocation failed")
```

## Configuration

### Output Control
```python
# Console only
logger_init(console_output=True, file_output=False)

# File only  
logger_init(console_output=False, file_output=True)

# Both (default)
logger_init(console_output=True, file_output=True)
```

### Log Levels
```python
# Debug mode (verbose)
logger_init(log_level="DEBUG")

# Production mode (minimal)
logger_init(log_level="WARNING")

# Custom log file
logger_init(log_file="custom.log")
```

## Log Levels

| Level | Usage | Example |
|-------|-------|---------|
| **DEBUG** | Detailed debugging info | `log.debug("Processing audio frame")` |
| **INFO** | General program flow | `log.info("Application started")` |
| **WARNING** | Expected issues/user actions | `log.warning("Window closed by user")` |
| **ERROR** | System failures/problems | `log.error("GPU memory allocation failed")` |
| **CRITICAL** | Program-breaking issues | `log.critical("OpenGL context lost")` |

## Exception Logging Best Practices

### WARNING Level (Expected/Controlled)
Use for user actions and normal program flow:
```python
# User closes window
log.warning("Window close detected")
raise WindowCloseException("Window Closed")

# End of audio file
log.warning("End of audio file reached")
raise EndOfAudioException("Audio file processing complete")

# User interrupts
log.warning("Keyboard Interrupt received.")
```

### ERROR Level (Unexpected/System Failures)
Use for system problems and resource issues:
```python
# GPU errors
except RuntimeError as e:
    log.error(f"GPU memory allocation failed: {e}")

# File I/O errors
except FileNotFoundError as e:
    log.error(f"Audio file not found: {e}")

# OpenGL errors
error = self.ctx.error
if error != 'GL_NO_ERROR':
    log.error(f"Render error: {error}")
```

## Decision Tree

```
Is the exception:
├── User-initiated action? → WARNING
├── Expected program flow? → WARNING  
├── Normal completion? → WARNING
├── System failure? → ERROR
├── Resource problem? → ERROR
└── Unknown/Unhandled? → ERROR
```

## Best Practices

1. **Use module names**: `get_logger(__name__)`
2. **Be specific**: Include context in messages
3. **Appropriate levels**: Don't log expected behavior as errors
4. **Performance**: Use `log.debug()` for expensive operations
5. **Cleanup**: Avoid logging in destructors (`__del__`) - use `print()` instead

## File Location

Logs are stored in `logs/subshader.log` by default. The directory is created automatically. 