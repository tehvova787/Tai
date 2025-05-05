"""
Caching Utilities for Lucky Train AI Assistant

This module provides caching functionality to improve performance by storing
frequently requested information and computed results.
"""

import logging
import time
from typing import Any, Dict, Optional, Union, Callable
import json
import os
import hashlib
import pickle
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacheEntry:
    """A class representing a cache entry with expiration."""
    
    def __init__(self, value: Any, ttl_seconds: int = 3600):
        """Initialize a cache entry.
        
        Args:
            value: The value to cache.
            ttl_seconds: Time to live in seconds.
        """
        self.value = value
        self.expiry_time = time.time() + ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired.
        
        Returns:
            True if expired, False otherwise.
        """
        return time.time() > self.expiry_time

class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize the memory cache.
        
        Args:
            max_size: Maximum number of entries in the cache.
            ttl_seconds: Default time to live in seconds.
        """
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = ttl_seconds
        logger.info(f"Initialized memory cache with max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The cache key.
            
        Returns:
            The cached value, or None if not found or expired.
        """
        entry = self.cache.get(key)
        
        if not entry:
            return None
        
        if entry.is_expired():
            # Remove expired entry
            del self.cache[key]
            return None
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Time to live in seconds. If None, use the default TTL.
        """
        # Use default TTL if not specified
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl
        
        # Create a new cache entry
        entry = CacheEntry(value, ttl_seconds)
        
        # Check if we need to evict an entry to make room
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Evict oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Cache full, evicted entry with key: {oldest_key}")
        
        # Add entry to cache
        self.cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key.
            
        Returns:
            True if the key was found and deleted, False otherwise.
        """
        if key in self.cache:
            del self.cache[key]
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
    
    def cleanup(self) -> int:
        """Remove all expired entries from the cache.
        
        Returns:
            The number of entries removed.
        """
        keys_to_delete = [key for key, entry in self.cache.items() if entry.is_expired()]
        
        for key in keys_to_delete:
            del self.cache[key]
        
        return len(keys_to_delete)

class DiskCache:
    """Disk-based cache implementation."""
    
    def __init__(self, cache_dir: str = ".cache", max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize the disk cache.
        
        Args:
            cache_dir: Directory to store cache files.
            max_size: Maximum number of entries in the cache.
            ttl_seconds: Default time to live in seconds.
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.default_ttl = ttl_seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized disk cache in {cache_dir} with max_size={max_size}, ttl={ttl_seconds}s")
    
    def _get_cache_file_path(self, key: str) -> str:
        """Get the file path for a cache key.
        
        Args:
            key: The cache key.
            
        Returns:
            The file path for the cache key.
        """
        # Hash the key to get a filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The cache key.
            
        Returns:
            The cached value, or None if not found or expired.
        """
        file_path = self._get_cache_file_path(key)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
            
            if entry.is_expired():
                # Remove expired entry
                os.remove(file_path)
                return None
            
            return entry.value
            
        except Exception as e:
            logger.error(f"Error reading cache file {file_path}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Time to live in seconds. If None, use the default TTL.
        """
        # Use default TTL if not specified
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl
        
        # Create a new cache entry
        entry = CacheEntry(value, ttl_seconds)
        
        # Save to disk
        file_path = self._get_cache_file_path(key)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.error(f"Error writing cache file {file_path}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key.
            
        Returns:
            True if the key was found and deleted, False otherwise.
        """
        file_path = self._get_cache_file_path(key)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                logger.error(f"Error deleting cache file {file_path}: {e}")
        
        return False
    
    def clear(self) -> None:
        """Clear the entire cache."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".cache"):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error deleting cache file {file_path}: {e}")
    
    def cleanup(self) -> int:
        """Remove all expired entries from the cache.
        
        Returns:
            The number of entries removed.
        """
        removed_count = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".cache"):
                file_path = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(file_path, 'rb') as f:
                        entry = pickle.load(f)
                    
                    if entry.is_expired():
                        os.remove(file_path)
                        removed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing cache file {file_path}: {e}")
        
        return removed_count

class CacheManager:
    """Manager for different cache implementations."""
    
    def __init__(self, config: Dict = None):
        """Initialize the cache manager.
        
        Args:
            config: Cache configuration dictionary.
        """
        self.config = config or {}
        
        # Get cache settings
        enabled = self.config.get("enabled", True)
        cache_type = self.config.get("type", "memory")
        ttl_seconds = self.config.get("ttl_seconds", 3600)
        max_size = self.config.get("max_size", 1000)
        cache_dir = self.config.get("cache_dir", ".cache")
        
        self.enabled = enabled
        
        # Initialize appropriate cache implementation
        if not enabled:
            self.cache = None
            logger.info("Caching is disabled")
            return
        
        if cache_type == "disk":
            self.cache = DiskCache(cache_dir, max_size, ttl_seconds)
        else:
            # Default to memory cache
            self.cache = MemoryCache(max_size, ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The cache key.
            
        Returns:
            The cached value, or None if not found, expired, or caching is disabled.
        """
        if not self.enabled or not self.cache:
            return None
        
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Time to live in seconds. If None, use the default TTL.
        """
        if not self.enabled or not self.cache:
            return
        
        self.cache.set(key, value, ttl_seconds)
    
    def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key.
            
        Returns:
            True if the key was found and deleted, False otherwise.
        """
        if not self.enabled or not self.cache:
            return False
        
        return self.cache.delete(key)
    
    def clear(self) -> None:
        """Clear the entire cache."""
        if not self.enabled or not self.cache:
            return
        
        self.cache.clear()
    
    def cleanup(self) -> int:
        """Remove all expired entries from the cache.
        
        Returns:
            The number of entries removed.
        """
        if not self.enabled or not self.cache:
            return 0
        
        return self.cache.cleanup()

def cached(manager: CacheManager, key_fn: Callable = None, ttl_seconds: Optional[int] = None):
    """Decorator to cache function results.
    
    Args:
        manager: The cache manager to use.
        key_fn: Function to generate a cache key from the function arguments.
            If None, use the function name and a hash of the arguments.
        ttl_seconds: Time to live in seconds. If None, use the default TTL.
        
    Returns:
        A decorator function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip caching if disabled
            if not manager.enabled:
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default key generation
                args_str = str(args) + str(sorted(kwargs.items()))
                args_hash = hashlib.md5(args_str.encode()).hexdigest()
                cache_key = f"{func.__name__}:{args_hash}"
            
            # Try to get from cache
            cached_result = manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result
            
            # Cache miss, compute result
            logger.debug(f"Cache miss for {cache_key}")
            result = func(*args, **kwargs)
            
            # Cache result
            manager.set(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    
    return decorator 