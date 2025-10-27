"""
Logging configuration for fantasy modeling system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = 'INFO',
                 log_file: Optional[str] = None,
                 log_dir: str = 'logs') -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name
        log_dir: Directory for log files

    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        log_file_path = log_path / log_file
    else:
        log_file_path = None

    # Create logger
    logger = logging.getLogger('fantasy_modeling')
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if log file specified
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger