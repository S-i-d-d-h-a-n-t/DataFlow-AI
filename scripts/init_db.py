"""
Utility script to initialise the database outside of the app lifecycle.
Usage: python scripts/init_db.py
"""

import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import setup_logging, get_logger
from app.core.database import create_all_tables

setup_logging()
logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Initialising database tables...")
    create_all_tables()
    logger.info("Done.")
