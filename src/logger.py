"""
Enhanced Logging Module for Lucky Train AI Assistant

This module provides advanced logging capabilities:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Log rotation
- Multiple output options (console, file, database)
- Structured logging with metadata
- JSON formatting option
"""

import logging
import logging.handlers
import os
import json
import time
import socket
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from functools import wraps
import threading
import queue
import uuid

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Default log file paths
DEFAULT_LOG_FILE = os.path.join(logs_dir, 'lucky_train_ai.log')
ERROR_LOG_FILE = os.path.join(logs_dir, 'error.log')
ACCESS_LOG_FILE = os.path.join(logs_dir, 'access.log')
SECURITY_LOG_FILE = os.path.join(logs_dir, 'security.log')

# Log record fields
STANDARD_FIELDS = [
    'timestamp', 'level', 'logger', 'message', 'module', 
    'function', 'line', 'thread', 'thread_name'
]

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_standard_fields: bool = True):
        """Initialize the JSON formatter.
        
        Args:
            include_standard_fields: Whether to include standard log fields
        """
        super().__init__()
        self.include_standard_fields = include_standard_fields
        self.hostname = socket.gethostname()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON string representation of the log record
        """
        # Create the log data dictionary
        log_data = {}
        
        # Add standard fields
        if self.include_standard_fields:
            log_data.update({
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread': record.thread,
                'thread_name': record.threadName,
                'process': record.process,
                'hostname': self.hostname
            })
        
        # Add any extra fields from the record
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)

class AsyncHandler(logging.Handler):
    """Asynchronous log handler for non-blocking logging."""
    
    def __init__(self, handler: logging.Handler):
        """Initialize the async handler.
        
        Args:
            handler: The underlying handler to use
        """
        super().__init__()
        self.handler = handler
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Add the record to the queue for processing.
        
        Args:
            record: The log record to emit
        """
        self.queue.put(record)
    
    def _process_queue(self) -> None:
        """Process records from the queue."""
        while True:
            try:
                record = self.queue.get()
                self.handler.emit(record)
                self.queue.task_done()
            except Exception:
                # Log the exception but keep the thread running
                sys.stderr.write("Error in async logger\n")
                traceback.print_exc(file=sys.stderr)

class DatabaseHandler(logging.Handler):
    """Log handler that writes to a database."""
    
    def __init__(self, db_connector: Any, table: str = 'logs'):
        """Initialize the database handler.
        
        Args:
            db_connector: The database connector to use
            table: The database table to write logs to
        """
        super().__init__()
        self.db_connector = db_connector
        self.table = table
    
    def emit(self, record: logging.LogRecord) -> None:
        """Write the log record to the database.
        
        Args:
            record: The log record to emit
        """
        try:
            # Format the record as a JSON string
            formatter = JSONFormatter()
            log_json = formatter.format(record)
            
            # Convert JSON to dict
            log_data = json.loads(log_json)
            
            # Prepare the query
            query = f"INSERT INTO {self.table} (timestamp, level, logger, message, data) VALUES (%s, %s, %s, %s, %s)"
            params = (
                log_data.get('timestamp'),
                log_data.get('level'),
                log_data.get('logger'),
                log_data.get('message'),
                json.dumps(log_data)
            )
            
            # Execute the query
            self.db_connector.execute_query(query, params)
        except Exception:
            # Don't raise exceptions from log handlers
            sys.stderr.write("Error in database logger\n")
            traceback.print_exc(file=sys.stderr)

class RequestIdFilter(logging.Filter):
    """Filter that adds a request ID to log records."""
    
    def __init__(self, request_id: Optional[str] = None):
        """Initialize the request ID filter.
        
        Args:
            request_id: Optional request ID to use
        """
        super().__init__()
        self.request_id = request_id or str(uuid.uuid4())
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add a request ID to the record.
        
        Args:
            record: The log record to filter
            
        Returns:
            True to include the record, False to exclude it
        """
        if not hasattr(record, 'extra'):
            record.extra = {}
        record.extra['request_id'] = self.request_id
        return True

class LuckyTrainLogger:
    """Enhanced logger for Lucky Train AI Assistant."""
    
    def __init__(self, name: str, config: Dict = None):
        """Initialize the logger.
        
        Args:
            name: Logger name
            config: Logger configuration dictionary
        """
        self.name = name
        self.config = config or {}
        
        # Get logger configuration
        self.log_level = self._get_log_level()
        self.log_file = self.config.get('log_file', DEFAULT_LOG_FILE)
        self.error_log_file = self.config.get('error_log_file', ERROR_LOG_FILE)
        self.access_log_file = self.config.get('access_log_file', ACCESS_LOG_FILE)
        self.security_log_file = self.config.get('security_log_file', SECURITY_LOG_FILE)
        self.max_bytes = self.config.get('max_bytes', 10 * 1024 * 1024)  # 10 MB
        self.backup_count = self.config.get('backup_count', 5)
        self.use_json = self.config.get('use_json', False)
        self.async_logging = self.config.get('async_logging', True)
        
        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Request ID for correlating log entries
        self.request_id_filter = RequestIdFilter()
        self.logger.addFilter(self.request_id_filter)
        
        # Add handlers
        self._add_console_handler()
        self._add_file_handler(self.log_file)
        self._add_file_handler(self.error_log_file, level=logging.ERROR)
        
        # Additional specialized loggers
        self.access_logger = self._create_specialized_logger('access', self.access_log_file)
        self.security_logger = self._create_specialized_logger('security', self.security_log_file)
        
        # Database logging is optional
        self.db_logger = None
        if self.config.get('db_logging', False):
            self._setup_db_logging()
    
    def _get_log_level(self) -> int:
        """Get the log level from the configuration.
        
        Returns:
            Log level as an integer
        """
        level_str = self.config.get('log_level', 'INFO').upper()
        return getattr(logging, level_str, logging.INFO)
    
    def _get_formatter(self) -> logging.Formatter:
        """Get a formatter based on the configuration.
        
        Returns:
            Configured formatter
        """
        if self.use_json:
            return JSONFormatter()
        else:
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
            )
    
    def _add_console_handler(self) -> None:
        """Add a console handler to the logger."""
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self, log_file: str, level: int = None) -> None:
        """Add a file handler to the logger.
        
        Args:
            log_file: Path to the log file
            level: Log level for this handler
        """
        # Create a rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        
        if level:
            file_handler.setLevel(level)
        
        file_handler.setFormatter(self._get_formatter())
        
        # Wrap with async handler if configured
        if self.async_logging:
            handler = AsyncHandler(file_handler)
        else:
            handler = file_handler
        
        self.logger.addHandler(handler)
    
    def _create_specialized_logger(self, type_name: str, log_file: str) -> logging.Logger:
        """Create a specialized logger for a specific purpose.
        
        Args:
            type_name: Type of logger to create
            log_file: Path to the log file
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(f"{self.name}.{type_name}")
        logger.setLevel(self.log_level)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add request ID filter
        logger.addFilter(self.request_id_filter)
        
        # Add file handler
        self._add_file_handler(log_file)
        
        return logger
    
    def _setup_db_logging(self) -> None:
        """Set up database logging if configured."""
        db_config = self.config.get('db_config', {})
        db_table = db_config.get('table', 'logs')
        
        try:
            # Import dynamically to avoid circular imports
            from database_connectors import create_db_connector
            
            db_type = db_config.get('type', 'sqlite')
            db_connector = create_db_connector(db_type, db_config)
            
            if db_connector and db_connector.connect():
                # Create the log table if it doesn't exist
                self._create_log_table(db_connector, db_table)
                
                # Create database handler
                db_handler = DatabaseHandler(db_connector, db_table)
                
                # Add to logger
                if self.async_logging:
                    self.logger.addHandler(AsyncHandler(db_handler))
                else:
                    self.logger.addHandler(db_handler)
                
                self.db_logger = db_connector
            else:
                self.logger.error("Failed to connect to database for logging")
        except Exception as e:
            # Don't fail if database logging can't be set up
            sys.stderr.write(f"Error setting up database logging: {e}\n")
    
    def _create_log_table(self, db_connector: Any, table: str) -> None:
        """Create the log table in the database if it doesn't exist.
        
        Args:
            db_connector: The database connector to use
            table: The table name to create
        """
        # This is a generic schema that should work for most SQL databases
        query = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            logger TEXT,
            message TEXT,
            data TEXT
        )
        """
        
        try:
            db_connector.execute_query(query)
        except Exception as e:
            sys.stderr.write(f"Error creating log table: {e}\n")
    
    def set_request_id(self, request_id: str) -> None:
        """Set the request ID for the current logging context.
        
        Args:
            request_id: The request ID to use
        """
        self.request_id_filter.request_id = request_id
    
    def debug(self, message: str, extra: Dict = None) -> None:
        """Log a debug message.
        
        Args:
            message: The message to log
            extra: Extra data to include with the log entry
        """
        self._log(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Dict = None) -> None:
        """Log an info message.
        
        Args:
            message: The message to log
            extra: Extra data to include with the log entry
        """
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Dict = None) -> None:
        """Log a warning message.
        
        Args:
            message: The message to log
            extra: Extra data to include with the log entry
        """
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Dict = None, exc_info: bool = False) -> None:
        """Log an error message.
        
        Args:
            message: The message to log
            extra: Extra data to include with the log entry
            exc_info: Whether to include exception info
        """
        self._log(logging.ERROR, message, extra, exc_info)
    
    def critical(self, message: str, extra: Dict = None, exc_info: bool = False) -> None:
        """Log a critical message.
        
        Args:
            message: The message to log
            extra: Extra data to include with the log entry
            exc_info: Whether to include exception info
        """
        self._log(logging.CRITICAL, message, extra, exc_info)
    
    def exception(self, message: str, extra: Dict = None) -> None:
        """Log an exception message.
        
        Args:
            message: The message to log
            extra: Extra data to include with the log entry
        """
        self._log(logging.ERROR, message, extra, exc_info=True)
    
    def _log(self, level: int, message: str, extra: Dict = None, exc_info: bool = False) -> None:
        """Log a message with the given level.
        
        Args:
            level: The log level
            message: The message to log
            extra: Extra data to include with the log entry
            exc_info: Whether to include exception info
        """
        # Prepare extra data
        log_extra = {}
        if extra:
            log_extra = extra.copy()
        
        # Add timestamp
        log_extra['timestamp'] = datetime.utcnow().isoformat()
        
        # Store extra data on the log record
        if not hasattr(self.logger, 'extra'):
            self.logger.extra = {}
        self.logger.extra = log_extra
        
        # Log the message
        self.logger.log(level, message, exc_info=exc_info, extra={'extra': log_extra})
    
    def log_access(self, user_id: str, endpoint: str, method: str, status_code: int, duration_ms: float, 
                  request_data: Dict = None, response_data: Dict = None) -> None:
        """Log an API access entry.
        
        Args:
            user_id: User identifier
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            request_data: Request data
            response_data: Response data
        """
        extra = {
            'user_id': user_id,
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'duration_ms': duration_ms,
            'request': request_data,
            'response': response_data
        }
        
        self.access_logger.info(f"Access: {method} {endpoint} - {status_code} ({duration_ms}ms)", extra=extra)
    
    def log_security(self, event_type: str, user_id: str, result: str, details: Dict = None) -> None:
        """Log a security event.
        
        Args:
            event_type: Type of security event
            user_id: User identifier
            result: Result of the event (success, failure, etc.)
            details: Additional event details
        """
        extra = {
            'event_type': event_type,
            'user_id': user_id,
            'result': result,
            'ip_address': details.get('ip_address') if details else None,
            'user_agent': details.get('user_agent') if details else None,
            'details': details
        }
        
        self.security_logger.info(f"Security: {event_type} - {result} - User: {user_id}", extra=extra)

def log_execution_time(logger: LuckyTrainLogger):
    """Decorator to log the execution time of a function.
    
    Args:
        logger: The logger to use
        
    Returns:
        The decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Log execution time
            logger.debug(
                f"Function {func.__name__} executed in {execution_time:.2f}ms",
                extra={
                    'function': func.__name__,
                    'execution_time_ms': execution_time
                }
            )
            
            return result
        return wrapper
    return decorator

# Default logger instance
def get_logger(name: str = 'lucky_train', config: Dict = None) -> LuckyTrainLogger:
    """Get or create a logger with the given name.
    
    Args:
        name: Logger name
        config: Logger configuration
        
    Returns:
        Configured logger
    """
    return LuckyTrainLogger(name, config) 