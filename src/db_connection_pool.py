"""
Database Connection Pool Module for Lucky Train AI Assistant

This module provides enhanced database connection pooling:
- Thread-safe connection management
- Connection timeouts and proper error handling
- Auto-recovery from connection failures
- Connection limits to prevent resource exhaustion
- Connection monitoring and logging
"""

import os
import logging
import sqlite3
import time
import threading
import queue
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from contextlib import contextmanager
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseConnectionError(Exception):
    """Exception raised for database connection errors."""
    pass

class ConnectionTimeoutError(DatabaseConnectionError):
    """Exception raised when a database connection times out."""
    pass

class PoolExhaustedError(DatabaseConnectionError):
    """Exception raised when the connection pool is exhausted."""
    pass

class PoolClosedError(DatabaseConnectionError):
    """Exception raised when trying to use a closed pool."""
    pass

class ConnectionInfo:
    """Information about a database connection."""
    
    def __init__(self, connection):
        """Initialize connection info.
        
        Args:
            connection: Database connection object
        """
        self.connection = connection
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.in_use = False
        self.use_count = 0
        self.transaction_count = 0
        self.last_error = None
        self.last_query = None
        self.thread_id = None
    
    def mark_used(self) -> None:
        """Mark the connection as being used."""
        self.last_used_at = time.time()
        self.in_use = True
        self.use_count += 1
        self.thread_id = threading.get_ident()
    
    def mark_free(self) -> None:
        """Mark the connection as free."""
        self.last_used_at = time.time()
        self.in_use = False
        self.thread_id = None
    
    def record_query(self, query: str) -> None:
        """Record a query executed on this connection.
        
        Args:
            query: SQL query executed
        """
        self.last_query = query
        self.last_used_at = time.time()
    
    def record_error(self, error: Exception) -> None:
        """Record an error that occurred on this connection.
        
        Args:
            error: Exception that occurred
        """
        self.last_error = {
            "error": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": time.time()
        }

class ConnectionPool:
    """Database connection pool."""
    
    def __init__(self, db_type: str, config: Dict = None):
        """Initialize the connection pool.
        
        Args:
            db_type: Database type (sqlite, mysql, postgres)
            config: Configuration dictionary
        """
        self.db_type = db_type
        self.config = config or {}
        
        # Pool configuration
        self.min_connections = max(1, self.config.get("min_connections", 2))
        self.max_connections = max(self.min_connections, self.config.get("max_connections", 10))
        self.max_overflow = self.config.get("max_overflow", 5)
        self.timeout = self.config.get("timeout", 30)  # seconds
        self.connection_lifetime = self.config.get("connection_lifetime", 3600)  # 1 hour
        self.idle_timeout = self.config.get("idle_timeout", 600)  # 10 minutes
        self.reconnect_delay = self.config.get("reconnect_delay", 5)  # seconds
        self.validation_interval = self.config.get("validation_interval", 60)  # seconds
        
        # Pool state
        self.pool = []  # List of ConnectionInfo objects
        self.lock = threading.RLock()
        self.closed = False
        self.last_validation = 0
        self.stats = {
            "created": 0,
            "closed": 0,
            "errors": 0,
            "timeouts": 0,
            "max_concurrent": 0,
            "total_wait_time": 0,
            "total_queries": 0
        }
        
        # Start with minimum connections
        self._initialize_pool()
        
        # Start maintenance thread
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name=f"ConnectionPool-{db_type}-Maintenance"
        )
        self.maintenance_thread.start()
        
        logger.info(f"Initialized {db_type} connection pool with {self.min_connections} "
                   f"to {self.max_connections}+{self.max_overflow} connections")
    
    def _initialize_pool(self) -> None:
        """Initialize the pool with the minimum number of connections."""
        with self.lock:
            for _ in range(self.min_connections):
                try:
                    connection = self._create_connection()
                    self.pool.append(ConnectionInfo(connection))
                    self.stats["created"] += 1
                except Exception as e:
                    logger.error(f"Error creating initial connection: {str(e)}")
    
    def _create_connection(self) -> Any:
        """Create a new database connection.
        
        Returns:
            Database connection object
            
        Raises:
            DatabaseConnectionError: If connection creation fails
        """
        try:
            if self.db_type == "sqlite":
                db_path = self.config.get("db_path", ":memory:")
                connection = sqlite3.connect(
                    db_path,
                    timeout=self.timeout,
                    isolation_level=None  # Autocommit mode
                )
                connection.row_factory = sqlite3.Row
                
                # Enable foreign keys
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys = ON")
                cursor.close()
                
                return connection
            
            elif self.db_type == "mysql":
                try:
                    import pymysql
                    connection = pymysql.connect(
                        host=self.config.get("host", "localhost"),
                        port=int(self.config.get("port", 3306)),
                        user=self.config.get("user", "root"),
                        password=self.config.get("password", ""),
                        database=self.config.get("database", ""),
                        charset=self.config.get("charset", "utf8mb4"),
                        connect_timeout=self.timeout,
                        autocommit=True,
                        cursorclass=pymysql.cursors.DictCursor
                    )
                    return connection
                except ImportError:
                    raise DatabaseConnectionError("pymysql package not installed")
            
            elif self.db_type == "postgres":
                try:
                    import psycopg2
                    import psycopg2.extras
                    
                    connection = psycopg2.connect(
                        host=self.config.get("host", "localhost"),
                        port=int(self.config.get("port", 5432)),
                        user=self.config.get("user", "postgres"),
                        password=self.config.get("password", ""),
                        dbname=self.config.get("database", ""),
                        connect_timeout=self.timeout,
                        application_name="LuckyTrainAI",
                        options="-c timezone=UTC"
                    )
                    connection.autocommit = True
                    return connection
                except ImportError:
                    raise DatabaseConnectionError("psycopg2 package not installed")
            
            else:
                raise DatabaseConnectionError(f"Unsupported database type: {self.db_type}")
                
        except Exception as e:
            # Wrap in our own exception
            self.stats["errors"] += 1
            raise DatabaseConnectionError(f"Failed to create {self.db_type} connection: {str(e)}")
    
    def _close_connection(self, conn_info: ConnectionInfo) -> None:
        """Close a database connection.
        
        Args:
            conn_info: Connection info object
        """
        try:
            conn_info.connection.close()
            self.stats["closed"] += 1
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
    
    def _validate_connection(self, conn_info: ConnectionInfo) -> bool:
        """Validate that a connection is still good.
        
        Args:
            conn_info: Connection info object
            
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Skip if connection is in use
            if conn_info.in_use:
                return True
                
            # Test with simple query
            if self.db_type == "sqlite":
                cursor = conn_info.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
            elif self.db_type == "mysql":
                with conn_info.connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            elif self.db_type == "postgres":
                with conn_info.connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            
            return True
        except Exception as e:
            logger.warning(f"Connection validation failed: {str(e)}")
            return False
    
    def _maintenance_loop(self) -> None:
        """Maintenance thread that performs periodic connection validation and cleanup."""
        while not self.closed:
            try:
                time.sleep(60)  # Check every minute
                
                if self.closed:
                    break
                    
                self._perform_maintenance()
                
            except Exception as e:
                logger.error(f"Error in connection pool maintenance: {str(e)}")
    
    def _perform_maintenance(self) -> None:
        """Perform maintenance tasks on the connection pool."""
        with self.lock:
            now = time.time()
            
            # Skip if recently validated
            if now - self.last_validation < self.validation_interval:
                return
                
            self.last_validation = now
                
            # Check each connection
            connections_to_remove = []
            
            for i, conn_info in enumerate(self.pool):
                # Skip connections in use
                if conn_info.in_use:
                    continue
                
                # Check if connection is too old
                too_old = now - conn_info.created_at > self.connection_lifetime
                
                # Check if connection has been idle too long
                idle_too_long = now - conn_info.last_used_at > self.idle_timeout
                
                # Check if we have too many connections
                too_many = (len(self.pool) - len(connections_to_remove) > self.min_connections and 
                           (too_old or idle_too_long))
                
                # Validate connections that aren't obviously expired
                invalid = False
                if not (too_old or idle_too_long) and not self._validate_connection(conn_info):
                    invalid = True
                
                # If any condition is met, close the connection
                if too_old or idle_too_long or too_many or invalid:
                    connections_to_remove.append(i)
            
            # Remove in reverse order to avoid index issues
            for i in sorted(connections_to_remove, reverse=True):
                if i < len(self.pool):  # Make sure index is still valid
                    conn_info = self.pool.pop(i)
                    self._close_connection(conn_info)
                    logger.debug(f"Closed {self.db_type} connection during maintenance: "
                               f"age={time.time() - conn_info.created_at:.1f}s, "
                               f"idle={time.time() - conn_info.last_used_at:.1f}s")
            
            # Ensure we still have minimum connections
            while len(self.pool) < self.min_connections:
                try:
                    connection = self._create_connection()
                    self.pool.append(ConnectionInfo(connection))
                    self.stats["created"] += 1
                except Exception as e:
                    logger.error(f"Failed to create connection during maintenance: {str(e)}")
                    # Don't retry immediately
                    break
            
            # Log pool stats
            active_count = sum(1 for conn in self.pool if conn.in_use)
            logger.debug(f"{self.db_type} pool stats: {len(self.pool)} connections, "
                        f"{active_count} active, {len(self.pool) - active_count} idle")
    
    def get_connection(self) -> Any:
        """Get a connection from the pool.
        
        Returns:
            Database connection object
            
        Raises:
            ConnectionTimeoutError: If timed out waiting for a connection
            PoolExhaustedError: If no connections available and can't create more
            PoolClosedError: If the pool is closed
        """
        if self.closed:
            raise PoolClosedError("Connection pool is closed")
        
        start_time = time.time()
        deadline = start_time + self.timeout
        
        while time.time() < deadline:
            with self.lock:
                # Try to find an unused connection
                for conn_info in self.pool:
                    if not conn_info.in_use:
                        # Mark as used and return
                        conn_info.mark_used()
                        
                        # Update stats
                        self.stats["total_wait_time"] += time.time() - start_time
                        active_count = sum(1 for c in self.pool if c.in_use)
                        self.stats["max_concurrent"] = max(self.stats["max_concurrent"], active_count)
                        
                        return conn_info.connection
                
                # If we have room, create a new connection
                if len(self.pool) < self.max_connections + self.max_overflow:
                    try:
                        connection = self._create_connection()
                        conn_info = ConnectionInfo(connection)
                        conn_info.mark_used()
                        self.pool.append(conn_info)
                        self.stats["created"] += 1
                        
                        # Update stats
                        self.stats["total_wait_time"] += time.time() - start_time
                        active_count = sum(1 for c in self.pool if c.in_use)
                        self.stats["max_concurrent"] = max(self.stats["max_concurrent"], active_count)
                        
                        return connection
                    except Exception as e:
                        logger.error(f"Error creating new connection: {str(e)}")
                        # Fall through to wait
            
            # Wait a bit and try again
            time.sleep(0.1)
        
        # Timed out
        self.stats["timeouts"] += 1
        raise ConnectionTimeoutError(f"Timed out waiting for a {self.db_type} connection")
    
    def return_connection(self, connection: Any) -> None:
        """Return a connection to the pool.
        
        Args:
            connection: Database connection to return
        """
        if self.closed:
            try:
                connection.close()
            except:
                pass
            return
        
        with self.lock:
            # Find connection in pool
            for conn_info in self.pool:
                if conn_info.connection is connection:
                    conn_info.mark_free()
                    return
            
            # If connection wasn't found, just close it
            try:
                connection.close()
                self.stats["closed"] += 1
            except:
                pass
    
    @contextmanager
    def connection(self):
        """Context manager for getting and releasing a connection.
        
        Yields:
            Database connection object
        """
        connection = None
        try:
            connection = self.get_connection()
            yield connection
        finally:
            if connection:
                self.return_connection(connection)
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self.lock:
            self.closed = True
            
            for conn_info in self.pool:
                try:
                    self._close_connection(conn_info)
                except:
                    pass
            
            self.pool = []
            
            logger.info(f"Closed all connections in {self.db_type} pool")
    
    def get_stats(self) -> Dict:
        """Get statistics about the connection pool.
        
        Returns:
            Dictionary of pool statistics
        """
        with self.lock:
            active_connections = sum(1 for conn in self.pool if conn.in_use)
            idle_connections = len(self.pool) - active_connections
            
            return {
                "db_type": self.db_type,
                "pool_size": len(self.pool),
                "active_connections": active_connections,
                "idle_connections": idle_connections,
                "min_connections": self.min_connections,
                "max_connections": self.max_connections,
                "max_overflow": self.max_overflow,
                "created_connections": self.stats["created"],
                "closed_connections": self.stats["closed"],
                "connection_errors": self.stats["errors"],
                "connection_timeouts": self.stats["timeouts"],
                "max_concurrent_connections": self.stats["max_concurrent"],
                "avg_wait_time": (self.stats["total_wait_time"] / (self.stats["created"] or 1)),
            }

class ConnectionPoolManager:
    """Manager for multiple database connection pools."""
    
    def __init__(self):
        """Initialize the connection pool manager."""
        self.pools = {}
        self.lock = threading.RLock()
    
    def get_pool(self, db_type: str, config: Dict = None) -> ConnectionPool:
        """Get or create a connection pool.
        
        Args:
            db_type: Database type (sqlite, mysql, postgres)
            config: Configuration dictionary
            
        Returns:
            Connection pool
        """
        # Create a key from db_type and connection parameters
        key_parts = [db_type]
        
        if config:
            if db_type == "sqlite":
                key_parts.append(config.get("db_path", ":memory:"))
            else:
                key_parts.extend([
                    config.get("host", "localhost"),
                    str(config.get("port", "")),
                    config.get("database", "")
                ])
        
        pool_key = ":".join(key_parts)
        
        with self.lock:
            if pool_key not in self.pools:
                self.pools[pool_key] = ConnectionPool(db_type, config)
            
            return self.pools[pool_key]
    
    def close_all(self) -> None:
        """Close all connection pools."""
        with self.lock:
            for pool in self.pools.values():
                pool.close_all()
            
            self.pools = {}
    
    def get_stats(self) -> Dict:
        """Get statistics about all connection pools.
        
        Returns:
            Dictionary of pool statistics
        """
        with self.lock:
            return {
                "pools": len(self.pools),
                "total_connections": sum(len(pool.pool) for pool in self.pools.values()),
                "active_connections": sum(sum(1 for conn in pool.pool if conn.in_use) 
                                       for pool in self.pools.values()),
                "pool_stats": {key: pool.get_stats() for key, pool in self.pools.items()}
            }

# Global connection pool manager instance
_pool_manager = None

def get_pool_manager() -> ConnectionPoolManager:
    """Get the connection pool manager instance.
    
    Returns:
        Connection pool manager
    """
    global _pool_manager
    
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    
    return _pool_manager

def get_connection_pool(db_type: str, config: Dict = None) -> ConnectionPool:
    """Get a connection pool from the manager.
    
    Args:
        db_type: Database type (sqlite, mysql, postgres)
        config: Configuration dictionary
        
    Returns:
        Connection pool
    """
    pool_manager = get_pool_manager()
    return pool_manager.get_pool(db_type, config)

@contextmanager
def get_connection(db_type: str, config: Dict = None):
    """Get a database connection from a pool.
    
    Args:
        db_type: Database type (sqlite, mysql, postgres)
        config: Configuration dictionary
        
    Yields:
        Database connection
    """
    pool = get_connection_pool(db_type, config)
    with pool.connection() as connection:
        yield connection

class ConnectionTracker:
    """Track database queries for debugging and performance monitoring."""
    
    def __init__(self, db_type: str, config: Dict = None):
        """Initialize the connection tracker.
        
        Args:
            db_type: Database type (sqlite, mysql, postgres)
            config: Configuration dictionary
        """
        self.db_type = db_type
        self.config = config or {}
        self.pool = get_connection_pool(db_type, config)
        self.stats = {
            "total_queries": 0,
            "total_time": 0,
            "slow_queries": 0,
            "error_queries": 0,
            "slow_threshold": self.config.get("slow_threshold", 1.0),  # seconds
            "queries": []
        }
        self.max_query_log = self.config.get("max_query_log", 100)
        self.lock = threading.RLock()
        self.enabled = self.config.get("enabled", True)
    
    @contextmanager
    def track_query(self, query: str, params: Any = None):
        """Track the execution of a database query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Yields:
            None
        """
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        error = None
        
        try:
            yield
        except Exception as e:
            error = e
            raise
        finally:
            duration = time.time() - start_time
            
            with self.lock:
                self.stats["total_queries"] += 1
                self.stats["total_time"] += duration
                
                if duration >= self.stats["slow_threshold"]:
                    self.stats["slow_queries"] += 1
                
                if error:
                    self.stats["error_queries"] += 1
                
                # Add to query log
                query_info = {
                    "query": query,
                    "params": params,
                    "duration": duration,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "error": str(error) if error else None,
                    "slow": duration >= self.stats["slow_threshold"]
                }
                
                self.stats["queries"].append(query_info)
                
                # Trim query log
                if len(self.stats["queries"]) > self.max_query_log:
                    self.stats["queries"] = self.stats["queries"][-self.max_query_log:]
                
                # Log slow queries
                if duration >= self.stats["slow_threshold"]:
                    logger.warning(f"Slow query ({duration:.3f}s): {query}")
    
    def get_stats(self) -> Dict:
        """Get query tracking statistics.
        
        Returns:
            Dictionary of tracking statistics
        """
        with self.lock:
            stats_copy = self.stats.copy()
            stats_copy["avg_query_time"] = (
                stats_copy["total_time"] / stats_copy["total_queries"]
                if stats_copy["total_queries"] > 0 else 0
            )
            return stats_copy
    
    def reset_stats(self) -> None:
        """Reset tracking statistics."""
        with self.lock:
            self.stats = {
                "total_queries": 0,
                "total_time": 0,
                "slow_queries": 0,
                "error_queries": 0,
                "slow_threshold": self.stats["slow_threshold"],
                "queries": []
            }
    
    def get_slow_queries(self) -> List[Dict]:
        """Get a list of slow queries.
        
        Returns:
            List of slow query dictionaries
        """
        with self.lock:
            return [q for q in self.stats["queries"] if q["slow"]]
    
    def get_error_queries(self) -> List[Dict]:
        """Get a list of queries that resulted in errors.
        
        Returns:
            List of error query dictionaries
        """
        with self.lock:
            return [q for q in self.stats["queries"] if q["error"]]

# Global query tracker instances
_query_trackers = {}

def get_query_tracker(db_type: str, config: Dict = None) -> ConnectionTracker:
    """Get a query tracker instance.
    
    Args:
        db_type: Database type (sqlite, mysql, postgres)
        config: Configuration dictionary
        
    Returns:
        Query tracker instance
    """
    global _query_trackers
    
    tracker_key = f"{db_type}:{config.get('db_path', '')}" if config else db_type
    
    if tracker_key not in _query_trackers:
        _query_trackers[tracker_key] = ConnectionTracker(db_type, config)
    
    return _query_trackers[tracker_key] 