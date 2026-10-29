"""Logging configuration."""
import logging
import sys

LOG_FORMAT = "%(levelname)-8s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Log messages of the bus_embeddings package to stdout."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger = logging.getLogger("bus_embeddings")
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    # Silence noisy third-party loggers unless debugging
    if log_level > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Logger in the bus_embeddings hierarchy."""
    if not name.startswith("bus_embeddings"):
        name = f"bus_embeddings.{name}"
    return logging.getLogger(name)
