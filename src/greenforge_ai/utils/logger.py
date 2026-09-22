# src/greenforge_ai/utils/logger.py
import logging
import sys
from pathlib import Path
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """
    Creates a dual-handler logger:
    - Terminal (Console): Simple, clean format for real-time monitoring.
    - File: Detailed enterprise format saved to the physical 'logs/' directory.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if initialized multiple times
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        # Resolve project root and set up logs directory
        project_root = Path(__file__).resolve().parents[3]
        log_dir = project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamped file log
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_file = log_dir / f"{timestamp}.txt"
        
        # 1. File Handler (Detailed Format for historical tracking)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 2. Console Handler (Simplified Format for Terminal visibility)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            fmt="%(levelname)s: %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger