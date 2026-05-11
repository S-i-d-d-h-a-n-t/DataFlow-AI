"""
Structured logging configuration for the application.
Uses Python's standard logging with JSON-friendly formatting.
"""

import logging
import sys
from app.core.config import settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured log lines."""

    def format(self, record: logging.LogRecord) -> str:
        record.app = settings.APP_NAME
        record.version = settings.APP_VERSION
        return super().format(record)


def setup_logging() -> None:
    """Configure root logger for the application."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = StructuredFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(app)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
