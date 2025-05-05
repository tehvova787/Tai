"""
System Improvements Integrator for Lucky Train AI Assistant

This module integrates all the improvements to address identified issues:
- Security and secrets management
- Memory monitoring and limiting
- Connection pooling and management
- Enhanced logging with proper rotation
- Thread safety improvements
"""

import os
import logging
import importlib
import json
from typing import Dict, List, Any, Optional, Union, Set, Callable
import threading
import traceback
import time
import sys
import signal

# Import improvement modules
from security_improvements import (
    SecretManager, 
    InputValidator,
    rotate_jwt_secret,
    verify_rotated_jwt
)

from memory_monitor import (
    MemoryMonitor, 
    MemoryStats, 
    MemoryLimitedDict, 
    get_memory_monitor
)

from logger_enhancements import (
    LogManager, 
    EnhancedLogger, 
    get_log_manager, 
    get_logger
)

from db_connection_pool import (
    ConnectionPool, 
    ConnectionPoolManager, 
    get_pool_manager, 
    get_connection_pool, 
    get_connection
)

# Configure basic logging until enhanced logging is set up
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemImprovement:
    """Base class for all system improvements."""
    
    def __init__(self, system_manager, config: Dict = None):
        """Initialize the system improvement.
        
        Args:
            system_manager: System manager instance
            config: Configuration dictionary
        """
        self.system_manager = system_manager
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.name = self.__class__.__name__
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the improvement.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.info(f"Improvement {self.name} is disabled")
            return False
            
        self.initialized = True
        logger.info(f"Initialized improvement: {self.name}")
        return True
    
    def shutdown(self) -> None:
        """Perform cleanup when shutting down."""
        pass

class SecurityImprovement(SystemImprovement):
    """Security improvements for the system."""
    
    def initialize(self) -> bool:
        """Initialize security improvements.
        
        Returns:
            True if successful, False otherwise
        """
        if not super().initialize():
            return False
        
        # Create secret manager
        self.secret_manager = SecretManager(self.config.get("secrets", {}))
        
        # Initialize validator
        self.validator = InputValidator()
        
        # Setup JWT key rotation if enabled
        jwt_config = self.config.get("jwt", {})
        if jwt_config.get("auto_rotate", False):
            rotation_days = jwt_config.get("rotation_days", 30)
            
            # Check if key needs rotation
            last_rotation = self.secret_manager.get("jwt_secret_rotation_time")
            if not last_rotation or (last_rotation and 
                                   time.time() - time.mktime(time.strptime(last_rotation, "%Y-%m-%dT%H:%M:%S.%f")) > rotation_days * 86400):
                # Rotate key
                rotate_jwt_secret(self.secret_manager)
        
        # Store in system manager for other components to use
        self.system_manager.secret_manager = self.secret_manager
        self.system_manager.input_validator = self.validator
        
        # Replace existing security manager if there is one
        if hasattr(self.system_manager, "security_manager"):
            # Update JWT verify method to use our improved version
            original_verify = self.system_manager.security_manager.verify_jwt_token
            
            def enhanced_verify_jwt(token):
                return verify_rotated_jwt(token, self.secret_manager)
            
            self.system_manager.security_manager.verify_jwt_token = enhanced_verify_jwt
            
            logger.info("Enhanced existing security manager with improved JWT verification")
        
        logger.info("Security improvements initialized")
        return True

class MemoryManagementImprovement(SystemImprovement):
    """Memory management improvements for the system."""
    
    def initialize(self) -> bool:
        """Initialize memory management improvements.
        
        Returns:
            True if successful, False otherwise
        """
        if not super().initialize():
            return False
        
        # Create memory monitor
        self.memory_monitor = get_memory_monitor(self.config)
        
        # Set up limits
        memory_limit_mb = self.config.get("memory_limit_mb", 1024)  # Default 1 GB
        self.memory_monitor.memory_limit_mb = memory_limit_mb
        
        # Register cleanup callbacks
        self._register_cleanup_callbacks()
        
        # Track large objects
        self._track_large_objects()
        
        # Start monitoring
        self.memory_monitor.start_monitoring()
        
        # Store in system manager for other components to use
        self.system_manager.memory_monitor = self.memory_monitor
        
        # Generate initial memory report
        self.memory_monitor.log_memory_report()
        
        logger.info(f"Memory management initialized with {memory_limit_mb}MB limit")
        return True
    
    def _register_cleanup_callbacks(self) -> None:
        """Register cleanup callbacks for memory management."""
        # Register warning-level cleanup (free caches, run garbage collection)
        self.memory_monitor.register_warning_callback(self._clean_caches)
        
        # Register critical-level cleanup (emergency measures)
        self.memory_monitor.register_critical_callback(self._emergency_cleanup)
    
    def _track_large_objects(self) -> None:
        """Track large objects in the system."""
        # Track vector database if available
        if hasattr(self.system_manager, "vector_db") and self.system_manager.vector_db:
            self.memory_monitor.track_object(self.system_manager.vector_db, "vector_db")
        
        # Track cache managers if available
        if hasattr(self.system_manager, "cache_manager") and self.system_manager.cache_manager:
            self.memory_monitor.track_object(self.system_manager.cache_manager, "cache_manager")
        
        if hasattr(self.system_manager, "question_cache") and self.system_manager.question_cache:
            self.memory_monitor.track_object(self.system_manager.question_cache, "question_cache")
    
    def _clean_caches(self) -> None:
        """Clean caches to free memory."""
        if hasattr(self.system_manager, "cache_manager") and self.system_manager.cache_manager:
            if hasattr(self.system_manager.cache_manager, "cleanup"):
                self.system_manager.cache_manager.cleanup()
        
        if hasattr(self.system_manager, "question_cache") and self.system_manager.question_cache:
            if hasattr(self.system_manager.question_cache, "cache") and hasattr(self.system_manager.question_cache.cache, "_clean_expired"):
                self.system_manager.question_cache.cache._clean_expired()
    
    def _emergency_cleanup(self) -> None:
        """Perform emergency cleanup to free memory."""
        # Clear all caches
        if hasattr(self.system_manager, "cache_manager") and self.system_manager.cache_manager:
            if hasattr(self.system_manager.cache_manager, "clear"):
                self.system_manager.cache_manager.clear()
        
        # Clear embedding cache if using vector DB
        if hasattr(self.system_manager, "vector_db") and self.system_manager.vector_db:
            if hasattr(self.system_manager.vector_db, "clear_cache"):
                self.system_manager.vector_db.clear_cache()
    
    def shutdown(self) -> None:
        """Stop memory monitoring."""
        if hasattr(self, "memory_monitor"):
            self.memory_monitor.log_memory_report()
            self.memory_monitor.stop_monitoring()

class LoggingImprovement(SystemImprovement):
    """Logging improvements for the system."""
    
    def initialize(self) -> bool:
        """Initialize logging improvements.
        
        Returns:
            True if successful, False otherwise
        """
        if not super().initialize():
            return False
        
        # Create log manager
        self.log_manager = get_log_manager(self.config)
        
        # Store in system manager for other components to use
        self.system_manager.log_manager = self.log_manager
        
        # Replace existing logger if there is one
        if hasattr(self.system_manager, "logger"):
            logger_name = self.system_manager.logger.name
            enhanced_logger = self.log_manager.get_logger(logger_name)
            self.system_manager.logger = enhanced_logger
            
            logger.info(f"Replaced existing logger '{logger_name}' with enhanced version")
        
        # Clean up old log files
        max_age_days = self.config.get("max_log_age_days", 30)
        removed_count = self.log_manager.cleanup_old_logs(max_age_days)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} log files older than {max_age_days} days")
        
        logger.info("Logging improvements initialized")
        return True

class DatabaseImprovement(SystemImprovement):
    """Database connection improvements for the system."""
    
    def initialize(self) -> bool:
        """Initialize database improvements.
        
        Returns:
            True if successful, False otherwise
        """
        if not super().initialize():
            return False
        
        # Create connection pool manager
        self.pool_manager = get_pool_manager()
        
        # Store in system manager for other components to use
        self.system_manager.db_pool_manager = self.pool_manager
        
        # Replace database connection method if there is one
        if hasattr(self.system_manager, "db") and self.system_manager.db:
            # Determine database type
            db_type = "sqlite"  # Default to SQLite
            
            if hasattr(self.system_manager.db, "db_path"):
                db_config = {"db_path": self.system_manager.db.db_path}
            else:
                db_config = {}
            
            # Create connection pool
            pool = self.pool_manager.get_pool(db_type, db_config)
            
            # Replace connection method if it exists
            if hasattr(self.system_manager.db, "get_connection"):
                original_get_connection = self.system_manager.db.get_connection
                
                @contextmanager
                def enhanced_get_connection():
                    with pool.connection() as connection:
                        yield connection
                
                self.system_manager.db.get_connection = enhanced_get_connection
                
                logger.info("Enhanced database connection handling with connection pooling")
        
        logger.info("Database improvements initialized")
        return True
    
    def shutdown(self) -> None:
        """Close all database connections."""
        if hasattr(self, "pool_manager"):
            self.pool_manager.close_all()

class ThreadSafetyImprovement(SystemImprovement):
    """Thread safety improvements for the system."""
    
    def initialize(self) -> bool:
        """Initialize thread safety improvements.
        
        Returns:
            True if successful, False otherwise
        """
        if not super().initialize():
            return False
        
        # Apply thread safety improvements to critical components
        self._enhance_caching()
        self._enhance_vector_db()
        
        # Create thread-safe dictionary wrappers for any unsafe collections
        self._enhance_collections()
        
        logger.info("Thread safety improvements initialized")
        return True
    
    def _enhance_caching(self) -> None:
        """Enhance caching with thread safety measures."""
        # Add locks to cache operations if needed
        if hasattr(self.system_manager, "cache_manager") and self.system_manager.cache_manager:
            if not hasattr(self.system_manager.cache_manager, "lock"):
                self.system_manager.cache_manager.lock = threading.RLock()
                
                # Wrap cache methods with lock
                if hasattr(self.system_manager.cache_manager, "get"):
                    original_get = self.system_manager.cache_manager.get
                    
                    def thread_safe_get(key):
                        with self.system_manager.cache_manager.lock:
                            return original_get(key)
                    
                    self.system_manager.cache_manager.get = thread_safe_get
                
                if hasattr(self.system_manager.cache_manager, "set"):
                    original_set = self.system_manager.cache_manager.set
                    
                    def thread_safe_set(key, value, ttl_seconds=None):
                        with self.system_manager.cache_manager.lock:
                            return original_set(key, value, ttl_seconds)
                    
                    self.system_manager.cache_manager.set = thread_safe_set
                
                if hasattr(self.system_manager.cache_manager, "delete"):
                    original_delete = self.system_manager.cache_manager.delete
                    
                    def thread_safe_delete(key):
                        with self.system_manager.cache_manager.lock:
                            return original_delete(key)
                    
                    self.system_manager.cache_manager.delete = thread_safe_delete
                
                logger.info("Enhanced cache manager with thread safety")
    
    def _enhance_vector_db(self) -> None:
        """Enhance vector database with thread safety measures."""
        # Add locks to vector database operations if needed
        if hasattr(self.system_manager, "vector_db") and self.system_manager.vector_db:
            if not hasattr(self.system_manager.vector_db, "lock"):
                self.system_manager.vector_db.lock = threading.RLock()
                
                # Wrap vector DB methods with lock
                if hasattr(self.system_manager.vector_db, "search"):
                    original_search = self.system_manager.vector_db.search
                    
                    def thread_safe_search(query, top_k=5, filter_criteria=None):
                        with self.system_manager.vector_db.lock:
                            return original_search(query, top_k, filter_criteria)
                    
                    self.system_manager.vector_db.search = thread_safe_search
                
                if hasattr(self.system_manager.vector_db, "_get_embedding"):
                    original_get_embedding = self.system_manager.vector_db._get_embedding
                    
                    def thread_safe_get_embedding(text):
                        with self.system_manager.vector_db.lock:
                            return original_get_embedding(text)
                    
                    self.system_manager.vector_db._get_embedding = thread_safe_get_embedding
                
                logger.info("Enhanced vector database with thread safety")
    
    def _enhance_collections(self) -> None:
        """Enhance collections with thread safety measures."""
        # Wrap non-thread-safe dictionaries with synchronized versions or locks
        # This is a general approach and might need customization based on specific cases
        pass

class ImprovementManager:
    """Manager for system improvements."""
    
    def __init__(self, system_manager, config_path: str = None):
        """Initialize the improvement manager.
        
        Args:
            system_manager: System manager instance
            config_path: Path to improvement configuration file
        """
        self.system_manager = system_manager
        self.improvements = []
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize improvements
        self._initialize_improvements()
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load improvement configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "security": {
                "enabled": True,
                "secrets": {
                    "auto_create": True,
                    "refresh_interval": 300  # 5 minutes
                },
                "jwt": {
                    "auto_rotate": True,
                    "rotation_days": 30
                }
            },
            "memory_management": {
                "enabled": True,
                "memory_limit_mb": 1024,  # 1 GB
                "warning_threshold": 0.8,  # 80%
                "critical_threshold": 0.95  # 95%
            },
            "logging": {
                "enabled": True,
                "max_log_age_days": 30,
                "max_bytes": 10 * 1024 * 1024,  # 10 MB
                "backup_count": 10,
                "compress_logs": True,
                "filter_sensitive": True
            },
            "database": {
                "enabled": True,
                "min_connections": 2,
                "max_connections": 10,
                "connection_lifetime": 3600,  # 1 hour
                "idle_timeout": 600  # 10 minutes
            },
            "thread_safety": {
                "enabled": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge configurations recursively
                self._merge_configs(default_config, user_config)
            except Exception as e:
                logger.error(f"Error loading improvement configuration: {e}")
        
        return default_config
    
    def _merge_configs(self, dest: Dict, src: Dict) -> None:
        """Merge configurations recursively.
        
        Args:
            dest: Destination configuration
            src: Source configuration
        """
        for key, value in src.items():
            if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
                self._merge_configs(dest[key], value)
            else:
                dest[key] = value
    
    def _initialize_improvements(self) -> None:
        """Initialize all improvements."""
        # Create and initialize improvements
        improvements = [
            SecurityImprovement(self.system_manager, self.config.get("security", {})),
            MemoryManagementImprovement(self.system_manager, self.config.get("memory_management", {})),
            LoggingImprovement(self.system_manager, self.config.get("logging", {})),
            DatabaseImprovement(self.system_manager, self.config.get("database", {})),
            ThreadSafetyImprovement(self.system_manager, self.config.get("thread_safety", {}))
        ]
        
        # Initialize each improvement
        for improvement in improvements:
            try:
                if improvement.initialize():
                    self.improvements.append(improvement)
            except Exception as e:
                logger.error(f"Failed to initialize improvement {improvement.name}: {e}")
                logger.error(traceback.format_exc())
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        # Define signal handler
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down improvements...")
            self.shutdown()
            
            # Call original system shutdown if exists
            if hasattr(self.system_manager, "shutdown"):
                try:
                    self.system_manager.shutdown()
                except Exception as e:
                    logger.error(f"Error in system shutdown: {e}")
            
            sys.exit(0)
        
        # Register handler for common signals
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self) -> None:
        """Shut down all improvements."""
        for improvement in reversed(self.improvements):
            try:
                logger.info(f"Shutting down {improvement.name}...")
                improvement.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {improvement.name}: {e}")

def enhance_system(system_manager, config_path: str = None) -> ImprovementManager:
    """Enhance a system with improvements.
    
    Args:
        system_manager: System manager instance
        config_path: Path to improvement configuration file
        
    Returns:
        Improvement manager
    """
    return ImprovementManager(system_manager, config_path)

if __name__ == "__main__":
    # This module is not meant to be run directly
    print("This module is meant to be imported, not run directly.") 