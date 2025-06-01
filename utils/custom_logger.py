import logging
import os
from datetime import datetime
from typing import Optional
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class CustomLogger:
    """
    Custom logger class that provides colored console output and clean log files
    """
    
    # Color mapping for different log levels
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        log_format: Optional[str] = None
    ):
        """
        Initialize the custom logger
        
        Args:
            name (str): Logger name
            log_file (str, optional): Path to log file
            level (int): Logging level
            log_format (str, optional): Custom log format
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not log_file:
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')

        if not log_format:
            log_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'

        # File handler (without color codes)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler (with color codes)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = self.ColoredFormatter(log_format)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    class ColoredFormatter(logging.Formatter):
        """
        Custom formatter class to add colors to console output
        """
        def format(self, record):
            # Save original values
            orig_levelname = record.levelname
            orig_msg = record.msg

            # Add color to level name and message
            color = CustomLogger.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
            record.msg = f"{color}{record.msg}{Style.RESET_ALL}"

            # Format the message
            result = super().format(record)

            # Restore original values
            record.levelname = orig_levelname
            record.msg = orig_msg
            
            return result

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)

# Usage example
if __name__ == "__main__":
    logger = CustomLogger("test_logger")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")