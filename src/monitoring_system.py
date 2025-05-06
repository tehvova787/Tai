"""
Monitoring System Module for Lucky Train AI

This module provides metrics collection and export in Prometheus format
for comprehensive monitoring of the Lucky Train AI system.

Key features:
- Performance metrics collection
- System health monitoring
- Request tracking and latency measurement
- Memory and resource usage tracking
- Prometheus metrics exposure
"""

import os
import time
import logging
import threading
# Try to import psutil, use a fallback if not available
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "psutil not installed. System metrics monitoring will be limited. "
        "Install with: pip install psutil"
    )
    # Create a mock psutil for basic functionality
    class MockProcess:
        @staticmethod
        def open_files():
            return []
    
    class MockPsutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 0.0
        
        @staticmethod
        def virtual_memory():
            class VirtualMemory:
                def __init__(self):
                    self.used = 1024 * 1024 * 1024  # Mock 1GB used
                    self.percent = 25.0  # Mock 25% used
            return VirtualMemory()
        
        @staticmethod
        def disk_usage(path):
            class DiskUsage:
                def __init__(self):
                    self.percent = 50.0  # Mock 50% used
            return DiskUsage()
        
        @staticmethod
        def Process(pid=None):
            return MockProcess()
    
    # Use the mock as a fallback
    psutil = MockPsutil()

import socket
import gc
from typing import Dict, List, Any, Optional, Union, Callable
# Try to import prometheus_client, use mock classes if not available
try:
    from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
    from prometheus_client.core import CollectorRegistry
    HAVE_PROMETHEUS = True
except ImportError:
    HAVE_PROMETHEUS = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "prometheus_client not installed. Metrics collection will be disabled. "
        "Install with: pip install prometheus-client"
    )
    # Create mock classes for Prometheus client
    class CollectorRegistry:
        def __init__(self):
            pass
    
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        
        def labels(self, **kwargs):
            return self
        
        def inc(self, amount=1):
            pass
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        
        def labels(self, **kwargs):
            return self
        
        def inc(self, amount=1):
            pass
        
        def dec(self, amount=1):
            pass
        
        def set(self, value):
            pass
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        
        def labels(self, **kwargs):
            return self
        
        def observe(self, value):
            pass
    
    class Summary:
        def __init__(self, *args, **kwargs):
            pass
        
        def labels(self, **kwargs):
            return self
        
        def observe(self, value):
            pass
    
    def start_http_server(port):
        logger.warning(f"Mock Prometheus HTTP server would start on port {port}")
        pass

from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and provides metrics for monitoring."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        # Create a registry for our metrics
        self.registry = CollectorRegistry()
        
        # System metrics
        self.cpu_usage = Gauge('lucky_train_cpu_usage_percent', 'CPU usage in percent', registry=self.registry)
        self.memory_usage = Gauge('lucky_train_memory_usage_bytes', 'Memory usage in bytes', registry=self.registry)
        self.memory_usage_percent = Gauge('lucky_train_memory_usage_percent', 'Memory usage in percent', registry=self.registry)
        self.disk_usage = Gauge('lucky_train_disk_usage_percent', 'Disk usage in percent', registry=self.registry)
        self.open_files = Gauge('lucky_train_open_files', 'Number of open files', registry=self.registry)
        self.thread_count = Gauge('lucky_train_thread_count', 'Number of active threads', registry=self.registry)
        
        # API metrics
        self.request_count = Counter('lucky_train_request_count_total', 'Total request count', 
                                  ['method', 'endpoint', 'status'], registry=self.registry)
        self.request_latency = Histogram('lucky_train_request_latency_seconds', 'Request latency in seconds',
                                      ['method', 'endpoint'], registry=self.registry)
        self.request_in_progress = Gauge('lucky_train_requests_in_progress', 'Requests currently in progress', 
                                      ['method', 'endpoint'], registry=self.registry)
        
        # AI model metrics
        self.model_request_count = Counter('lucky_train_model_request_count_total', 'Total AI model request count', 
                                        ['model_type', 'model_name'], registry=self.registry)
        self.model_latency = Histogram('lucky_train_model_latency_seconds', 'AI model latency in seconds',
                                     ['model_type', 'model_name'], registry=self.registry,
                                     buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0))
        self.token_count = Counter('lucky_train_token_count_total', 'Total token count', 
                                ['model_type', 'model_name', 'direction'], registry=self.registry)
        self.model_errors = Counter('lucky_train_model_errors_total', 'AI model errors', 
                                 ['model_type', 'model_name', 'error_type'], registry=self.registry)
        
        # Vector DB metrics
        self.vector_search_count = Counter('lucky_train_vector_search_count_total', 'Vector search count', 
                                        ['db_type'], registry=self.registry)
        self.vector_search_latency = Histogram('lucky_train_vector_search_latency_seconds', 'Vector search latency in seconds',
                                            ['db_type'], registry=self.registry)
        self.vector_insert_count = Counter('lucky_train_vector_insert_count_total', 'Vector insert count', 
                                        ['db_type'], registry=self.registry)
        self.vector_document_count = Gauge('lucky_train_vector_document_count', 'Number of documents in the vector DB', 
                                        ['db_type', 'collection'], registry=self.registry)
        
        # Cache metrics
        self.cache_hit_count = Counter('lucky_train_cache_hit_count_total', 'Cache hit count', 
                                    ['cache_type'], registry=self.registry)
        self.cache_miss_count = Counter('lucky_train_cache_miss_count_total', 'Cache miss count', 
                                     ['cache_type'], registry=self.registry)
        self.cache_size = Gauge('lucky_train_cache_size', 'Cache size', 
                             ['cache_type'], registry=self.registry)
        
        # Error metrics
        self.error_count = Counter('lucky_train_error_count_total', 'Error count', 
                                ['module', 'error_type'], registry=self.registry)
        
        # Init metrics collection thread
        self.should_run = True
        self.collector_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self.hostname = socket.gethostname()
        
        logger.info("Metrics collector initialized")
    
    def start_collector(self, interval: int = 15) -> None:
        """Start the metrics collector thread.
        
        Args:
            interval: Collection interval in seconds
        """
        self.collection_interval = interval
        self.collector_thread.start()
        logger.info(f"Metrics collector started with {interval}s interval")
    
    def stop_collector(self) -> None:
        """Stop the metrics collector thread."""
        self.should_run = False
        self.collector_thread.join(timeout=5.0)
        logger.info("Metrics collector stopped")
    
    def _collect_system_metrics(self) -> None:
        """Collect system metrics periodically."""
        while self.should_run:
            try:
                # CPU usage
                self.cpu_usage.set(psutil.cpu_percent(interval=1))
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                self.memory_usage_percent.set(memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.disk_usage.set(disk.percent)
                
                # Open files
                self.open_files.set(len(psutil.Process().open_files()))
                
                # Thread count
                self.thread_count.set(threading.active_count())
                
                # Wait for next collection
                time.sleep(self.collection_interval)
            
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(self.collection_interval)
    
    @contextmanager
    def track_request_time(self, method: str, endpoint: str) -> None:
        """Track request time using a context manager.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
        """
        self.request_in_progress.labels(method=method, endpoint=endpoint).inc()
        start_time = time.time()
        status = "success"
        
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.request_latency.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)
            self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
            self.request_in_progress.labels(method=method, endpoint=endpoint).dec()
    
    @contextmanager
    def track_model_time(self, model_type: str, model_name: str) -> None:
        """Track AI model latency using a context manager.
        
        Args:
            model_type: Model type (cloud, local, hybrid)
            model_name: Model name
        """
        start_time = time.time()
        error_type = "none"
        
        try:
            yield
        except Exception as e:
            error_type = type(e).__name__
            self.model_errors.labels(model_type=model_type, model_name=model_name, error_type=error_type).inc()
            raise
        finally:
            self.model_latency.labels(model_type=model_type, model_name=model_name).observe(time.time() - start_time)
            self.model_request_count.labels(model_type=model_type, model_name=model_name).inc()
    
    @contextmanager
    def track_vector_search_time(self, db_type: str) -> None:
        """Track vector database search latency using a context manager.
        
        Args:
            db_type: Database type
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            self.vector_search_latency.labels(db_type=db_type).observe(time.time() - start_time)
            self.vector_search_count.labels(db_type=db_type).inc()
    
    def record_token_count(self, model_type: str, model_name: str, direction: str, count: int) -> None:
        """Record token count for an AI model.
        
        Args:
            model_type: Model type (cloud, local, hybrid)
            model_name: Model name
            direction: Direction (input, output)
            count: Token count
        """
        self.token_count.labels(model_type=model_type, model_name=model_name, direction=direction).inc(count)
    
    def record_cache_hit(self, cache_type: str) -> None:
        """Record a cache hit.
        
        Args:
            cache_type: Cache type
        """
        self.cache_hit_count.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str) -> None:
        """Record a cache miss.
        
        Args:
            cache_type: Cache type
        """
        self.cache_miss_count.labels(cache_type=cache_type).inc()
    
    def set_cache_size(self, cache_type: str, size: int) -> None:
        """Set cache size.
        
        Args:
            cache_type: Cache type
            size: Cache size
        """
        self.cache_size.labels(cache_type=cache_type).set(size)
    
    def record_vector_insert(self, db_type: str, count: int = 1) -> None:
        """Record a vector database insert.
        
        Args:
            db_type: Database type
            count: Number of vectors inserted
        """
        self.vector_insert_count.labels(db_type=db_type).inc(count)
    
    def set_vector_document_count(self, db_type: str, collection: str, count: int) -> None:
        """Set vector database document count.
        
        Args:
            db_type: Database type
            collection: Collection name
            count: Document count
        """
        self.vector_document_count.labels(db_type=db_type, collection=collection).set(count)
    
    def record_error(self, module: str, error_type: str) -> None:
        """Record an error.
        
        Args:
            module: Module name
            error_type: Error type
        """
        self.error_count.labels(module=module, error_type=error_type).inc()

class MonitoringService:
    """Service for system monitoring and metrics exposure."""
    
    def __init__(self, config: Dict = None):
        """Initialize the monitoring service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.metrics_port = self.config.get("metrics_port", 9090)
        self.collection_interval = self.config.get("collection_interval", 15)
        self.enabled = self.config.get("enabled", True)
        
        # Create metrics collector
        self.metrics = MetricsCollector()
        self.server_started = False
        
        logger.info(f"Monitoring service initialized with port {self.metrics_port}")
    
    def start(self) -> None:
        """Start the monitoring service."""
        if not self.enabled:
            logger.info("Monitoring service is disabled, not starting")
            return
        
        try:
            # Start metrics HTTP server
            start_http_server(self.metrics_port)
            self.server_started = True
            
            # Start system metrics collector
            self.metrics.start_collector(self.collection_interval)
            
            logger.info(f"Prometheus metrics exposed on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start monitoring service: {e}")
    
    def stop(self) -> None:
        """Stop the monitoring service."""
        if self.enabled and self.metrics:
            self.metrics.stop_collector()
            logger.info("Monitoring service stopped")

class MetricsMiddleware:
    """Middleware for tracking API requests in frameworks like Flask or FastAPI."""
    
    def __init__(self, metrics: MetricsCollector):
        """Initialize the metrics middleware.
        
        Args:
            metrics: Metrics collector instance
        """
        self.metrics = metrics
        logger.info("Metrics middleware initialized")
    
    def flask_before_request(self) -> None:
        """Flask before request handler."""
        from flask import request, g
        
        g.start_time = time.time()
    
    def flask_after_request(self, response):
        """Flask after request handler.
        
        Args:
            response: Flask response
            
        Returns:
            Flask response
        """
        from flask import request, g
        
        endpoint = request.endpoint or request.path
        method = request.method
        status = response.status_code
        
        # Record request count
        self.metrics.request_count.labels(
            method=method,
            endpoint=endpoint,
            status="success" if status < 400 else "error"
        ).inc()
        
        # Record request latency
        if hasattr(g, 'start_time'):
            self.metrics.request_latency.labels(
                method=method,
                endpoint=endpoint
            ).observe(time.time() - g.start_time)
        
        return response
    
    def flask_register(self, app) -> None:
        """Register middleware with a Flask app.
        
        Args:
            app: Flask app
        """
        app.before_request(self.flask_before_request)
        app.after_request(self.flask_after_request)
        logger.info("Registered metrics middleware with Flask app")
    
    async def fastapi_middleware(self, request, call_next):
        """FastAPI middleware.
        
        Args:
            request: FastAPI request
            call_next: Next middleware function
            
        Returns:
            FastAPI response
        """
        method = request.method
        endpoint = request.url.path
        
        # Track request time
        with self.metrics.track_request_time(method, endpoint):
            response = await call_next(request)
        
        return response
    
    def fastapi_register(self, app) -> None:
        """Register middleware with a FastAPI app.
        
        Args:
            app: FastAPI app
        """
        from fastapi import FastAPI
        from starlette.middleware.base import BaseHTTPMiddleware
        
        class PrometheusMiddleware(BaseHTTPMiddleware):
            def __init__(self, app, metrics_middleware):
                super().__init__(app)
                self.metrics_middleware = metrics_middleware
            
            async def dispatch(self, request, call_next):
                return await self.metrics_middleware.fastapi_middleware(request, call_next)
        
        app.add_middleware(PrometheusMiddleware, metrics_middleware=self)
        logger.info("Registered metrics middleware with FastAPI app")

# Singleton instance
_monitoring_service = None

def get_monitoring_service(config: Dict = None) -> MonitoringService:
    """Get the monitoring service instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Monitoring service instance
    """
    global _monitoring_service
    
    if _monitoring_service is None:
        _monitoring_service = MonitoringService(config)
    
    return _monitoring_service

def get_metrics() -> MetricsCollector:
    """Get the metrics collector instance.
    
    Returns:
        Metrics collector instance
    """
    service = get_monitoring_service()
    return service.metrics

class MonitoringSystem:
    """Unified monitoring system for Lucky Train AI.
    
    This class provides a unified interface for monitoring all aspects of the system,
    including performance, memory usage, API calls, and model performance.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the monitoring system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get("monitoring_enabled", True)
        self.service = get_monitoring_service(config)
        self.metrics = self.service.metrics
        self.started = False
        
        logger.info("Monitoring system initialized")
    
    def start(self) -> None:
        """Start the monitoring system."""
        if not self.enabled:
            logger.info("Monitoring system is disabled")
            return
        
        if self.started:
            logger.info("Monitoring system already started")
            return
        
        self.service.start()
        self.started = True
    
    def stop(self) -> None:
        """Stop the monitoring system."""
        if not self.enabled or not self.started:
            return
        
        self.service.stop()
        self.started = False
    
    def track_request(self, method: str, endpoint: str) -> contextmanager:
        """Get a context manager for tracking request time.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            
        Returns:
            Context manager for tracking request time
        """
        return self.metrics.track_request_time(method, endpoint)
    
    def track_model(self, model_type: str, model_name: str) -> contextmanager:
        """Get a context manager for tracking model latency.
        
        Args:
            model_type: Model type (cloud, local, hybrid)
            model_name: Model name
            
        Returns:
            Context manager for tracking model latency
        """
        return self.metrics.track_model_time(model_type, model_name)
    
    def track_vector_search(self, db_type: str) -> contextmanager:
        """Get a context manager for tracking vector search latency.
        
        Args:
            db_type: Database type
            
        Returns:
            Context manager for tracking vector search latency
        """
        return self.metrics.track_vector_search_time(db_type)
    
    def record_tokens(self, model_type: str, model_name: str, input_tokens: int, output_tokens: int) -> None:
        """Record token counts for an AI model.
        
        Args:
            model_type: Model type (cloud, local, hybrid)
            model_name: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self.metrics.record_token_count(model_type, model_name, "input", input_tokens)
        self.metrics.record_token_count(model_type, model_name, "output", output_tokens)
    
    def record_cache_hit(self, cache_type: str) -> None:
        """Record a cache hit.
        
        Args:
            cache_type: Cache type
        """
        self.metrics.record_cache_hit(cache_type)
    
    def record_cache_miss(self, cache_type: str) -> None:
        """Record a cache miss.
        
        Args:
            cache_type: Cache type
        """
        self.metrics.record_cache_miss(cache_type)
    
    def update_cache_size(self, cache_type: str, size: int) -> None:
        """Update cache size.
        
        Args:
            cache_type: Cache type
            size: Cache size
        """
        self.metrics.set_cache_size(cache_type, size)
    
    def record_error(self, module: str, error_type: str) -> None:
        """Record an error.
        
        Args:
            module: Module name
            error_type: Error type
        """
        self.metrics.record_error(module, error_type)
    
    def get_stats(self) -> Dict:
        """Get system statistics.
        
        Returns:
            Dictionary of statistics
        """
        # Get basic stats from Prometheus registry
        if not self.enabled:
            return {"monitoring_enabled": False}
        
        stats = {
            "monitoring_enabled": True,
            "monitoring_started": self.started,
            "metrics_port": self.service.metrics_port,
            "collection_interval": self.service.collection_interval
        }
        
        return stats

# Get a singleton instance
def get_monitoring_system(config: Dict = None) -> MonitoringSystem:
    """Get the monitoring system instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Monitoring system instance
    """
    return MonitoringSystem(config) 