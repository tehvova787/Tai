"""
Logger Enhancements for Lucky Train AI Assistant

This module provides enhanced logging capabilities:
- Consistent log levels across components
- Size-limited log rotation with compression
- Log filtering for sensitive information
- Log aggregation
- Structured JSON logging
- Log search capabilities
"""

import os
import logging
import logging.handlers
import json
import time
import datetime
import re
import gzip
import shutil
import glob
from typing import Dict, List, Any, Optional, Union, Set
from functools import wraps
from threading import Lock

# Base directory for logs
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Default log files
DEFAULT_LOG_FILE = os.path.join(LOGS_DIR, 'lucky_train_ai.log')
ERROR_LOG_FILE = os.path.join(LOGS_DIR, 'error.log')
DEBUG_LOG_FILE = os.path.join(LOGS_DIR, 'debug.log')
ACCESS_LOG_FILE = os.path.join(LOGS_DIR, 'access.log')
SECURITY_LOG_FILE = os.path.join(LOGS_DIR, 'security.log')

# Log config
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 10
DEFAULT_LOG_LEVEL = logging.INFO

# Sensitive patterns to filter out (e.g., API keys, passwords)
SENSITIVE_PATTERNS = [
    (r'api[_-]?key\s*[=:]\s*[\'"](.*?)[\'"]', 'api_key=*****'),
    (r'password\s*[=:]\s*[\'"](.*?)[\'"]', 'password=*****'),
    (r'secret\s*[=:]\s*[\'"](.*?)[\'"]', 'secret=*****'),
    (r'Authorization: Bearer (.*?)\b', 'Authorization: Bearer *****'),
    (r'Authorization: (.*?)\b', 'Authorization: *****'),
    (r'token\s*[=:]\s*[\'"](.*?)[\'"]', 'token=*****')
]

# Thread lock for log file operations
log_file_lock = Lock()

class SensitiveFilter(logging.Filter):
    """Filter for removing sensitive information from logs."""
    
    def __init__(self, patterns: List[tuple] = None):
        """Initialize the sensitive filter.
        
        Args:
            patterns: List of (regex_pattern, replacement) tuples
        """
        super().__init__()
        self.patterns = patterns or SENSITIVE_PATTERNS
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), replacement) 
                                 for pattern, replacement in self.patterns]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive information from log records.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to include the record (always)
        """
        # Filter message
        if isinstance(record.msg, str):
            for pattern, replacement in self.compiled_patterns:
                record.msg = pattern.sub(replacement, record.msg)
        
        # Filter args if they are strings
        if record.args:
            args_list = list(record.args)
            for i, arg in enumerate(args_list):
                if isinstance(arg, str):
                    for pattern, replacement in self.compiled_patterns:
                        args_list[i] = pattern.sub(replacement, arg)
            record.args = tuple(args_list)
        
        return True

class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating file handler that compresses rotated logs."""
    
    def __init__(self, filename: str, mode: str = 'a', maxBytes: int = 0,
                backupCount: int = 0, encoding: str = None, delay: bool = False,
                compress: bool = True):
        """Initialize the compressed rotating file handler.
        
        Args:
            filename: Log file path
            mode: File mode
            maxBytes: Maximum file size in bytes
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Whether to delay opening the file
            compress: Whether to compress rotated logs
        """
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress = compress
    
    def doRollover(self):
        """Compress logs when rotating."""
        with log_file_lock:
            # Close the file
            if self.stream:
                self.stream.close()
                self.stream = None
            
            # Rotate the files
            if self.backupCount > 0:
                # Remove the oldest log if it exists
                oldest_log = f"{self.baseFilename}.{self.backupCount}"
                oldest_compressed = f"{oldest_log}.gz"
                if os.path.exists(oldest_compressed):
                    os.remove(oldest_compressed)
                if os.path.exists(oldest_log):
                    os.remove(oldest_log)
                
                # Shift the other logs
                for i in range(self.backupCount - 1, 0, -1):
                    source = f"{self.baseFilename}.{i}"
                    source_compressed = f"{source}.gz"
                    dest = f"{self.baseFilename}.{i + 1}"
                    dest_compressed = f"{dest}.gz"
                    
                    if os.path.exists(source_compressed):
                        if os.path.exists(dest_compressed):
                            os.remove(dest_compressed)
                        os.rename(source_compressed, dest_compressed)
                    elif os.path.exists(source):
                        if os.path.exists(dest_compressed):
                            os.remove(dest_compressed)
                        if self.compress:
                            with open(source, 'rb') as f_in:
                                with gzip.open(dest_compressed, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            os.remove(source)
                        else:
                            if os.path.exists(dest):
                                os.remove(dest)
                            os.rename(source, dest)
                
                # Rename the current log
                if os.path.exists(self.baseFilename):
                    dest = f"{self.baseFilename}.1"
                    dest_compressed = f"{dest}.gz"
                    if os.path.exists(dest_compressed):
                        os.remove(dest_compressed)
                    if self.compress:
                        with open(self.baseFilename, 'rb') as f_in:
                            with gzip.open(dest_compressed, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    else:
                        if os.path.exists(dest):
                            os.remove(dest)
                        os.rename(self.baseFilename, dest)
            
            # Open a new log file
            self.mode = 'w'
            self.stream = self._open()

class JsonFormatter(logging.Formatter):
    """Format logs as JSON objects."""
    
    def __init__(self, include_timestamp: bool = True, extra_fields: Dict = None):
        """Initialize the JSON formatter.
        
        Args:
            include_timestamp: Whether to include a timestamp field
            extra_fields: Extra fields to include in all logs
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.extra_fields = extra_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": self.formatMessage(record),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # Add timestamp if requested
        if self.include_timestamp:
            log_data["timestamp"] = datetime.datetime.fromtimestamp(
                record.created
            ).isoformat()
        
        # Add extra fields
        log_data.update(self.extra_fields)
        
        # Add any extra attributes from the record
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_data)

class EnhancedLogger:
    """Enhanced logger with additional capabilities."""
    
    def __init__(self, name: str, config: Dict = None):
        """Initialize the enhanced logger.
        
        Args:
            name: Logger name
            config: Logger configuration
        """
        self.name = name
        self.config = config or {}
        
        # Get configuration
        self.log_level = self._get_log_level()
        self.max_bytes = self.config.get('max_bytes', DEFAULT_MAX_BYTES)
        self.backup_count = self.config.get('backup_count', DEFAULT_BACKUP_COUNT)
        self.log_file = self.config.get('log_file', DEFAULT_LOG_FILE)
        self.error_log_file = self.config.get('error_log_file', ERROR_LOG_FILE)
        self.debug_log_file = self.config.get('debug_log_file', DEBUG_LOG_FILE)
        self.compress_logs = self.config.get('compress_logs', True)
        self.use_json = self.config.get('use_json', False)
        self.filter_sensitive = self.config.get('filter_sensitive', True)
        self.console_output = self.config.get('console_output', True)
        
        # Get or create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add handlers
        self._add_handlers()
    
    def _get_log_level(self) -> int:
        """Get log level from configuration.
        
        Returns:
            Log level as an integer
        """
        level_name = self.config.get('log_level', 'INFO').upper()
        return getattr(logging, level_name, DEFAULT_LOG_LEVEL)
    
    def _add_handlers(self) -> None:
        """Add handlers to the logger."""
        # Add console handler if enabled
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_formatter = self._get_formatter()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Add main log file handler
        main_handler = self._create_file_handler(
            self.log_file, 
            self.log_level
        )
        self.logger.addHandler(main_handler)
        
        # Add error log file handler
        error_handler = self._create_file_handler(
            self.error_log_file, 
            logging.ERROR
        )
        self.logger.addHandler(error_handler)
        
        # Add debug log file handler if debug enabled
        if self.log_level <= logging.DEBUG:
            debug_handler = self._create_file_handler(
                self.debug_log_file, 
                logging.DEBUG
            )
            self.logger.addHandler(debug_handler)
        
        # Add sensitive information filter if enabled
        if self.filter_sensitive:
            sensitive_filter = SensitiveFilter()
            self.logger.addFilter(sensitive_filter)
    
    def _create_file_handler(self, filename: str, level: int) -> logging.Handler:
        """Create a file handler.
        
        Args:
            filename: Log file path
            level: Log level for this handler
            
        Returns:
            Configured log handler
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create handler
        handler = CompressedRotatingFileHandler(
            filename=filename,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            compress=self.compress_logs
        )
        
        handler.setLevel(level)
        handler.setFormatter(self._get_formatter())
        
        return handler
    
    def _get_formatter(self) -> logging.Formatter:
        """Get a log formatter.
        
        Returns:
            Configured formatter
        """
        if self.use_json:
            return JsonFormatter(
                include_timestamp=True,
                extra_fields={"app": "lucky_train_ai"}
            )
        else:
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                '%Y-%m-%d %H:%M:%S'
            )
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message.
        
        Args:
            message: Message to log
            **kwargs: Additional fields to include
        """
        if kwargs:
            self.logger.debug(message, extra={"extra": kwargs})
        else:
            self.logger.debug(message)
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message.
        
        Args:
            message: Message to log
            **kwargs: Additional fields to include
        """
        if kwargs:
            self.logger.info(message, extra={"extra": kwargs})
        else:
            self.logger.info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message.
        
        Args:
            message: Message to log
            **kwargs: Additional fields to include
        """
        if kwargs:
            self.logger.warning(message, extra={"extra": kwargs})
        else:
            self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log an error message.
        
        Args:
            message: Message to log
            exc_info: Whether to include exception info
            **kwargs: Additional fields to include
        """
        if kwargs:
            self.logger.error(message, exc_info=exc_info, extra={"extra": kwargs})
        else:
            self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log a critical message.
        
        Args:
            message: Message to log
            exc_info: Whether to include exception info
            **kwargs: Additional fields to include
        """
        if kwargs:
            self.logger.critical(message, exc_info=exc_info, extra={"extra": kwargs})
        else:
            self.logger.critical(message, exc_info=exc_info)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log an exception message.
        
        Args:
            message: Message to log
            **kwargs: Additional fields to include
        """
        if kwargs:
            self.logger.exception(message, extra={"extra": kwargs})
        else:
            self.logger.exception(message)

class LogManager:
    """Manager for multiple loggers."""
    
    def __init__(self, config: Dict = None):
        """Initialize the log manager.
        
        Args:
            config: Log configuration
        """
        self.config = config or {}
        self.loggers = {}
        self.default_config = {
            'log_level': self.config.get('log_level', 'INFO'),
            'max_bytes': self.config.get('max_bytes', DEFAULT_MAX_BYTES),
            'backup_count': self.config.get('backup_count', DEFAULT_BACKUP_COUNT),
            'compress_logs': self.config.get('compress_logs', True),
            'use_json': self.config.get('use_json', False),
            'filter_sensitive': self.config.get('filter_sensitive', True),
            'console_output': self.config.get('console_output', True),
        }
    
    def get_logger(self, name: str, logger_config: Dict = None) -> EnhancedLogger:
        """Get or create a logger.
        
        Args:
            name: Logger name
            logger_config: Configuration for this specific logger
            
        Returns:
            Enhanced logger instance
        """
        if name in self.loggers:
            return self.loggers[name]
        
        # Merge configurations
        config = self.default_config.copy()
        if logger_config:
            config.update(logger_config)
        
        # Create logger
        logger = EnhancedLogger(name, config)
        self.loggers[name] = logger
        
        return logger
    
    def cleanup_old_logs(self, max_age_days: int = 30) -> int:
        """Remove log files older than max_age_days.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of files removed
        """
        with log_file_lock:
            # Get all log files
            log_files = []
            for pattern in ['*.log', '*.log.[0-9]', '*.log.[0-9].gz', '*.log.[0-9][0-9].gz']:
                log_files.extend(glob.glob(os.path.join(LOGS_DIR, pattern)))
            
            # Calculate cutoff time
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
            
            # Delete old files
            removed_count = 0
            for log_file in log_files:
                try:
                    if os.path.getmtime(log_file) < cutoff_time:
                        os.remove(log_file)
                        removed_count += 1
                except (OSError, IOError) as e:
                    print(f"Error removing old log file {log_file}: {e}")
            
            return removed_count
    
    def set_global_log_level(self, level: Union[str, int]) -> None:
        """Set the log level for all loggers.
        
        Args:
            level: Log level name or number
        """
        # Convert string to level number if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)
        
        # Update default config
        self.default_config['log_level'] = level
        
        # Update existing loggers
        for logger in self.loggers.values():
            logger.logger.setLevel(level)
            for handler in logger.logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(level)

# Global log manager instance
_log_manager = None

def get_log_manager(config: Dict = None) -> LogManager:
    """Get the log manager instance.
    
    Args:
        config: Log manager configuration
        
    Returns:
        Log manager instance
    """
    global _log_manager
    
    if _log_manager is None:
        _log_manager = LogManager(config)
    
    return _log_manager

def get_logger(name: str, config: Dict = None) -> EnhancedLogger:
    """Get a logger from the log manager.
    
    Args:
        name: Logger name
        config: Logger configuration
        
    Returns:
        Enhanced logger instance
    """
    log_manager = get_log_manager()
    return log_manager.get_logger(name, config)

def log_execution_time(logger: Optional[EnhancedLogger] = None, level: str = "DEBUG"):
    """Decorator to log function execution time.
    
    Args:
        logger: Logger to use, or None to get default
        level: Log level name
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger if not provided
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            # Set log level function
            log_func = getattr(logger, level.lower())
            
            # Time the function
            start_time = time.time()
            result = func(*args, **kwargs)
            exec_time = time.time() - start_time
            
            # Log execution time
            log_func(
                f"Function {func.__name__} executed in {exec_time:.6f} seconds",
                function=func.__name__,
                execution_time=exec_time
            )
            
            return result
        return wrapper
    return decorator 