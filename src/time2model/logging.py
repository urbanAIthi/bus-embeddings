"""Centralized logging configuration for time2model."""
import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FORMAT_SIMPLE = "%(levelname)-8s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    level: str = "INFO",
    simple_format: bool = True,
) -> None:
    """Configure logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger for time2model
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    formatter = logging.Formatter(
        LOG_FORMAT_SIMPLE if simple_format else LOG_FORMAT,
        datefmt=DATE_FORMAT
    )
    handler.setFormatter(formatter)
    
    # Configure time2model logger hierarchy
    logger = logging.getLogger("time2model")
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    
    # Suppress noisy third-party loggers unless in debug mode
    if log_level > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the time2model hierarchy."""
    # Ensure logger is under time2model hierarchy
    if not name.startswith("time2model"):
        name = f"time2model.{name}"
    return logging.getLogger(name)
