"""
Memory Monitoring Module for Lucky Train AI Assistant

This module provides memory monitoring capabilities:
- Track memory usage of the application
- Set memory limits and cleanup policies
- Monitor large data structures (caches, vector stores)
- Provide warnings and trigger cleanup when approaching limits
"""

import os
import sys
import logging
import time
import threading
import gc
from typing import Dict, List, Any, Optional, Callable, Union
# Try to import psutil, use a fallback if not available
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "psutil not installed. Memory monitoring will be limited. "
        "Install with: pip install psutil"
    )
    # Create a mock psutil for basic functionality
    class MockProcess:
        def __init__(self, pid):
            self.pid = pid
        
        def memory_info(self):
            class MemInfo:
                def __init__(self):
                    self.rss = 50 * 1024 * 1024  # Mock 50MB RSS
                    self.vms = 100 * 1024 * 1024  # Mock 100MB VMS
            return MemInfo()
    
    class MockVirtualMemory:
        def __init__(self):
            self.total = 4 * 1024 * 1024 * 1024  # Mock 4GB total
            self.available = 2 * 1024 * 1024 * 1024  # Mock 2GB available
            self.used = 2 * 1024 * 1024 * 1024  # Mock 2GB used
            self.percent = 50.0  # Mock 50% used
    
    class MockPsutil:
        @staticmethod
        def Process(pid):
            return MockProcess(pid)
        
        @staticmethod
        def virtual_memory():
            return MockVirtualMemory()
    
    # Use the mock as a fallback
    psutil = MockPsutil()

import weakref
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "numpy not installed. Memory monitoring for numpy arrays will be limited. "
        "Install with: pip install numpy"
    )
    # Create a simple numpy mock
    class MockNumpy:
        class ndarray:
            def __init__(self, *args, **kwargs):
                self.nbytes = 1024 * 1024  # Mock 1MB
    
    # Use the mock as a fallback
    np = MockNumpy()

from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryStats:
    """Memory statistics collector."""
    
    def __init__(self):
        """Initialize the memory statistics collector."""
        self.process = psutil.Process(os.getpid())
    
    def get_process_memory(self) -> Dict[str, float]:
        """Get memory usage for the current process.
        
        Returns:
            Dictionary with memory usage in MB
        """
        try:
            mem_info = self.process.memory_info()
            return {
                "rss": mem_info.rss / (1024 * 1024),  # Resident Set Size in MB
                "vms": mem_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
            }
        except Exception as e:
            logger.error(f"Error getting process memory: {e}")
            return {"rss": 0, "vms": 0}
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get system memory usage.
        
        Returns:
            Dictionary with memory usage in MB and percentages
        """
        try:
            mem = psutil.virtual_memory()
            return {
                "total": mem.total / (1024 * 1024),  # Total memory in MB
                "available": mem.available / (1024 * 1024),  # Available memory in MB
                "used": mem.used / (1024 * 1024),  # Used memory in MB
                "percent": mem.percent  # Percentage used
            }
        except Exception as e:
            logger.error(f"Error getting system memory: {e}")
            return {"total": 0, "available": 0, "used": 0, "percent": 0}
    
    def get_object_size(self, obj: Any) -> float:
        """Get the approximate memory size of an object in MB.
        
        Args:
            obj: The object to measure
            
        Returns:
            Size in MB
        """
        import sys
        
        # Special handling for numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.nbytes / (1024 * 1024)
        
        # For dictionaries, handle key-value pairs
        if isinstance(obj, dict):
            size = sys.getsizeof(obj)
            for key, value in obj.items():
                size += sys.getsizeof(key)
                if isinstance(value, (dict, list, set, np.ndarray)):
                    size += self.get_object_size(value) * (1024 * 1024)  # Convert back to bytes
                else:
                    size += sys.getsizeof(value)
            return size / (1024 * 1024)
        
        # For lists and sets, handle items
        if isinstance(obj, (list, set, tuple)):
            size = sys.getsizeof(obj)
            for item in obj:
                if isinstance(item, (dict, list, set, np.ndarray)):
                    size += self.get_object_size(item) * (1024 * 1024)  # Convert back to bytes
                else:
                    size += sys.getsizeof(item)
            return size / (1024 * 1024)
        
        # For other objects, use sys.getsizeof
        return sys.getsizeof(obj) / (1024 * 1024)

class MemoryMonitor:
    """Memory usage monitor and manager."""
    
    def __init__(self, config: Dict = None):
        """Initialize the memory monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Memory limits
        self.memory_limit_mb = self.config.get("memory_limit_mb", 1024)  # 1 GB default
        self.warning_threshold = self.config.get("warning_threshold", 0.8)  # 80% default
        self.critical_threshold = self.config.get("critical_threshold", 0.95)  # 95% default
        
        # Monitoring interval
        self.monitoring_interval = self.config.get("monitoring_interval", 60)  # 1 minute default
        
        # Callbacks for cleanup
        self.warning_callbacks = []
        self.critical_callbacks = []
        
        # Memory stats
        self.stats = MemoryStats()
        
        # Tracking references (using weak references to avoid memory leaks)
        self.tracked_objects = {}  # name -> weakref
        
        # Thread for monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Last memory usage
        self.last_memory_usage = {
            "process": {"rss": 0, "vms": 0},
            "system": {"total": 0, "available": 0, "used": 0, "percent": 0},
            "tracked_objects": {},
            "timestamp": time.time()
        }
    
    def start_monitoring(self) -> None:
        """Start the memory monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the memory monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        
        logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread."""
        while self.monitoring_active:
            try:
                self._check_memory_usage()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(10)  # Wait a bit before retrying
    
    def _check_memory_usage(self) -> None:
        """Check current memory usage and trigger callbacks if needed."""
        # Get memory stats
        process_memory = self.stats.get_process_memory()
        system_memory = self.stats.get_system_memory()
        
        # Check tracked objects
        tracked_sizes = {}
        for name, ref in list(self.tracked_objects.items()):
            obj = ref()
            if obj is None:
                # Object was garbage collected
                del self.tracked_objects[name]
                continue
            
            tracked_sizes[name] = self.stats.get_object_size(obj)
        
        # Update last memory usage
        self.last_memory_usage = {
            "process": process_memory,
            "system": system_memory,
            "tracked_objects": tracked_sizes,
            "timestamp": time.time()
        }
        
        # Check against limits
        process_rss_mb = process_memory["rss"]
        
        if process_rss_mb >= self.memory_limit_mb * self.critical_threshold:
            logger.warning(f"CRITICAL: Memory usage at {process_rss_mb:.2f} MB, "
                          f"{process_rss_mb / self.memory_limit_mb * 100:.1f}% of limit")
            self._trigger_critical_cleanup()
        
        elif process_rss_mb >= self.memory_limit_mb * self.warning_threshold:
            logger.info(f"WARNING: Memory usage at {process_rss_mb:.2f} MB, "
                       f"{process_rss_mb / self.memory_limit_mb * 100:.1f}% of limit")
            self._trigger_warning_cleanup()
    
    def _trigger_warning_cleanup(self) -> None:
        """Trigger warning level cleanup callbacks."""
        for callback, args, kwargs in self.warning_callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in warning cleanup callback: {e}")
    
    def _trigger_critical_cleanup(self) -> None:
        """Trigger critical level cleanup callbacks."""
        # Trigger warning callbacks first
        self._trigger_warning_cleanup()
        
        # Then trigger critical callbacks
        for callback, args, kwargs in self.critical_callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in critical cleanup callback: {e}")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Forced garbage collection: {collected} objects collected")
    
    def register_warning_callback(self, callback: Callable, *args, **kwargs) -> None:
        """Register a callback for warning level memory usage.
        
        Args:
            callback: Function to call
            args: Positional arguments for the callback
            kwargs: Keyword arguments for the callback
        """
        self.warning_callbacks.append((callback, args, kwargs))
    
    def register_critical_callback(self, callback: Callable, *args, **kwargs) -> None:
        """Register a callback for critical level memory usage.
        
        Args:
            callback: Function to call
            args: Positional arguments for the callback
            kwargs: Keyword arguments for the callback
        """
        self.critical_callbacks.append((callback, args, kwargs))
    
    def track_object(self, obj: Any, name: str) -> None:
        """Track memory usage of an object.
        
        Args:
            obj: Object to track
            name: Name for the tracked object
        """
        if not obj:
            return
        
        # Store as weak reference to avoid keeping the object alive
        self.tracked_objects[name] = weakref.ref(obj)
        
        # Log initial size
        size_mb = self.stats.get_object_size(obj)
        logger.info(f"Started tracking object '{name}' with size {size_mb:.2f} MB")
    
    def stop_tracking(self, name: str) -> None:
        """Stop tracking an object.
        
        Args:
            name: Name of the tracked object
        """
        if name in self.tracked_objects:
            del self.tracked_objects[name]
    
    def get_memory_report(self) -> Dict:
        """Get a comprehensive memory usage report.
        
        Returns:
            Dictionary with memory usage information
        """
        # Get current stats
        process_memory = self.stats.get_process_memory()
        system_memory = self.stats.get_system_memory()
        
        # Get tracked object sizes
        tracked_sizes = {}
        for name, ref in list(self.tracked_objects.items()):
            obj = ref()
            if obj is None:
                continue
            
            tracked_sizes[name] = self.stats.get_object_size(obj)
        
        # Sorted tracked objects by size (largest first)
        sorted_tracked = sorted(
            tracked_sizes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate change from last check
        process_change = {
            "rss": process_memory["rss"] - self.last_memory_usage["process"]["rss"],
            "vms": process_memory["vms"] - self.last_memory_usage["process"]["vms"]
        }
        
        # Build report
        report = {
            "timestamp": time.time(),
            "process_memory_mb": process_memory,
            "system_memory_mb": system_memory,
            "memory_limit_mb": self.memory_limit_mb,
            "usage_percent": process_memory["rss"] / self.memory_limit_mb * 100,
            "change_since_last_mb": process_change,
            "tracked_objects": {name: size for name, size in sorted_tracked},
            "warning_threshold_percent": self.warning_threshold * 100,
            "critical_threshold_percent": self.critical_threshold * 100
        }
        
        return report
    
    def log_memory_report(self) -> None:
        """Log a memory usage report."""
        report = self.get_memory_report()
        
        logger.info(f"=== Memory Usage Report ===")
        logger.info(f"Process RSS: {report['process_memory_mb']['rss']:.2f} MB")
        logger.info(f"Process VMS: {report['process_memory_mb']['vms']:.2f} MB")
        logger.info(f"Memory limit: {report['memory_limit_mb']:.2f} MB")
        logger.info(f"Usage: {report['usage_percent']:.1f}% of limit")
        
        logger.info(f"Change since last check: RSS {report['change_since_last_mb']['rss']:.2f} MB, "
                  f"VMS {report['change_since_last_mb']['vms']:.2f} MB")
        
        if report['tracked_objects']:
            logger.info(f"Top tracked objects by size:")
            for i, (name, size) in enumerate(report['tracked_objects'].items()):
                if i >= 5:  # Show only top 5
                    break
                logger.info(f"  - {name}: {size:.2f} MB")
        
        logger.info(f"System memory: {report['system_memory_mb']['percent']:.1f}% used "
                   f"({report['system_memory_mb']['used']:.0f} MB / {report['system_memory_mb']['total']:.0f} MB)")

def memory_limit(limit_mb: Optional[float] = None):
    """Decorator to limit memory usage of a function.
    
    Args:
        limit_mb: Memory limit in MB, or None to use global limit
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get memory stats
            stats = MemoryStats()
            process_memory_before = stats.get_process_memory()
            
            # Call function
            result = func(*args, **kwargs)
            
            # Check memory usage
            process_memory_after = stats.get_process_memory()
            memory_used = process_memory_after["rss"] - process_memory_before["rss"]
            
            # Log memory usage
            logger.debug(f"Function {func.__name__} used {memory_used:.2f} MB of memory")
            
            # Check limit
            if limit_mb is not None and memory_used > limit_mb:
                logger.warning(f"Function {func.__name__} exceeded memory limit: "
                              f"{memory_used:.2f} MB used, {limit_mb:.2f} MB limit")
            
            return result
        
        return wrapper
    
    return decorator

# Example usage for caches and other objects that need memory monitoring
class MemoryLimitedDict(dict):
    """Dictionary with memory usage monitoring and limits."""
    
    def __init__(self, max_memory_mb: float = 100, *args, **kwargs):
        """Initialize the memory-limited dictionary.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            args: Positional arguments for dict constructor
            kwargs: Keyword arguments for dict constructor
        """
        super().__init__(*args, **kwargs)
        self.max_memory_mb = max_memory_mb
        self.stats = MemoryStats()
        self.current_memory_mb = 0
        self.update_memory_usage()
    
    def update_memory_usage(self) -> float:
        """Update and return the current memory usage.
        
        Returns:
            Memory usage in MB
        """
        self.current_memory_mb = self.stats.get_object_size(self)
        return self.current_memory_mb
    
    def __setitem__(self, key, value):
        """Set item with memory usage check.
        
        Args:
            key: Dictionary key
            value: Value to set
        """
        # Calculate size of the new item
        temp_dict = {key: value}
        item_size = self.stats.get_object_size(temp_dict)
        
        # Check if we need to free up space
        if self.current_memory_mb + item_size > self.max_memory_mb:
            self._free_memory(item_size)
        
        # Set the item
        super().__setitem__(key, value)
        
        # Update memory usage
        self.update_memory_usage()
    
    def _free_memory(self, needed_mb: float) -> None:
        """Free up memory by removing items.
        
        Args:
            needed_mb: Amount of memory to free in MB
        """
        # Update current usage
        self.update_memory_usage()
        
        # If still empty enough, do nothing
        if self.current_memory_mb + needed_mb <= self.max_memory_mb:
            return
        
        # Sort items by (approximated) size
        items_sizes = []
        for key, value in list(self.items()):
            temp_dict = {key: value}
            size = self.stats.get_object_size(temp_dict)
            items_sizes.append((key, size))
        
        # Sort by size (smallest first)
        items_sizes.sort(key=lambda x: x[1])
        
        # Remove items until we have enough free space
        removed = 0
        for key, size in items_sizes:
            del self[key]
            removed += 1
            
            # Update memory usage
            self.update_memory_usage()
            
            # Check if we have enough free space
            if self.current_memory_mb + needed_mb <= self.max_memory_mb:
                break
        
        logger.info(f"Memory-limited dict removed {removed} items to free up space")

# Global memory monitor instance
_memory_monitor = None

def get_memory_monitor(config: Dict = None) -> MemoryMonitor:
    """Get the memory monitor instance.
    
    Args:
        config: Memory monitor configuration
        
    Returns:
        Memory monitor instance
    """
    global _memory_monitor
    
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor(config)
    
    return _memory_monitor 