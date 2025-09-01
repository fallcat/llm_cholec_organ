"""Logging utilities for endopoint."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "endopoint",
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string (uses default if None)
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str = "endopoint") -> logging.Logger:
    """Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)