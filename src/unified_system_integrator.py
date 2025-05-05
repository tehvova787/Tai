"""
Lucky Train AI - Unified System Integrator

This module integrates all system components into a unified architecture,
implementing the microservices design pattern while maintaining compatibility
with the existing codebase.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Union
import threading
import time
from pathlib import Path

# Import improved system components
from improved_system_init import init_system, get_system
from system_improvements import ImprovementManager
from security_improvements import SecretManager, InputValidator
from memory_monitor import MemoryMonitor
from logger_enhancements import configure_enhanced_logging
from db_connection_pool import ConnectionPool
from enhanced_vector_db import EnhancedVectorDB
from monitoring_system import MonitoringSystem

class UnifiedSystem:
    """
    Main class that integrates all system components into a unified architecture.
    Implements the façade pattern to provide a simple interface to the complex subsystem.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the unified system.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.initialized = False
        self.services = {}
        self.service_threads = {}
        self.monitoring = None
        self.logger = logging.getLogger("UnifiedSystem")
        
        # Initialize the system with improvements
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        try:
            # Initialize base system
            self.system = init_system(self.config_path)
            
            # Configure enhanced logging
            self.logger = configure_enhanced_logging("UnifiedSystem", self.system.config)
            self.logger.info("Initializing Unified System...")
            
            # Setup improvement manager
            self.improvement_manager = ImprovementManager(self.system.config)
            
            # Initialize security manager
            self.secret_manager = SecretManager()
            self.input_validator = InputValidator()
            
            # Initialize memory monitor
            self.memory_monitor = MemoryMonitor(
                warning_threshold=self.system.config.get("memory_management", {}).get("warning_threshold", 0.8),
                critical_threshold=self.system.config.get("memory_management", {}).get("critical_threshold", 0.95)
            )
            
            # Initialize connection pool for database
            self.db_pool = ConnectionPool(
                self.system.config.get("database", {})
            )
            
            # Initialize enhanced vector database
            self.vector_db = EnhancedVectorDB(
                self.system.config.get("vector_db", {}),
                self.db_pool
            )
            
            # Initialize monitoring system
            self.monitoring = MonitoringSystem(self.system.config)
            
            # Initialize all services
            self._initialize_services()
            
            self.initialized = True
            self.logger.info("Unified System initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Unified System: {e}")
            raise
    
    def _initialize_services(self):
        """Initialize all microservices."""
        self.logger.info("Initializing microservices...")
        
        # Import microservices
        from web_interface import LuckyTrainWebInterface
        from bot.telegram_bot import LuckyTrainTelegramBot
        from blockchain_integration import TONBlockchainIntegration
        from metaverse_integration import LuckyTrainMetaverseAssistant
        from ai_models import AIModelManager
        from multimodal_interface import MultimodalInterface
        
        # Initialize API Gateway Service
        self.services["api_gateway"] = self._create_api_gateway()
        
        # Initialize Assistant Core Service
        from bot.assistant import LuckyTrainAssistant
        self.services["assistant_core"] = LuckyTrainAssistant(self.config_path)
        
        # Initialize Knowledge Base Service
        self.services["knowledge_base"] = self.vector_db
        
        # Initialize AI Model Service
        self.services["ai_model"] = AIModelManager(self.system.config)
        
        # Initialize Blockchain Integration Service
        self.services["blockchain"] = TONBlockchainIntegration(self.config_path)
        
        # Initialize User Management Service
        self.services["user_management"] = self._create_user_management()
        
        # Initialize Metaverse Connector Service
        self.services["metaverse"] = LuckyTrainMetaverseAssistant(self.config_path)
        
        # Initialize Web Interface
        self.services["web_interface"] = LuckyTrainWebInterface(self.config_path)
        
        # Initialize Telegram Bot
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if token:
            self.services["telegram_bot"] = LuckyTrainTelegramBot(token, self.config_path)
        else:
            self.logger.warning("TELEGRAM_BOT_TOKEN not set, Telegram Bot service not initialized")
        
        # Initialize Multimodal Interface
        self.services["multimodal"] = MultimodalInterface(self.config_path)
        
        self.logger.info(f"Initialized {len(self.services)} microservices")
    
    def _create_api_gateway(self):
        """Create the API Gateway service."""
        # This would be implemented with a proper API gateway in production
        # For now, we're using a simple implementation
        class APIGatewayService:
            def __init__(self, system_config):
                self.config = system_config
                self.logger = logging.getLogger("APIGateway")
                self.routes = {}
                self.logger.info("API Gateway initialized")
            
            def register_route(self, path, service, method):
                self.routes[path] = (service, method)
                self.logger.info(f"Registered route: {path}")
            
            def handle_request(self, path, data):
                if path in self.routes:
                    service, method = self.routes[path]
                    return getattr(service, method)(data)
                else:
                    return {"error": "Route not found"}
        
        return APIGatewayService(self.system.config)
    
    def _create_user_management(self):
        """Create the User Management service."""
        # Implement a basic user management service
        class UserManagementService:
            def __init__(self, db_pool):
                self.db_pool = db_pool
                self.logger = logging.getLogger("UserManagement")
                self.logger.info("User Management service initialized")
                self.users = {}  # In-memory user store for demo
            
            def get_user(self, user_id):
                return self.users.get(user_id, {"id": user_id, "preferences": {}})
            
            def update_user(self, user_id, data):
                if user_id not in self.users:
                    self.users[user_id] = {"id": user_id, "preferences": {}}
                self.users[user_id].update(data)
                return True
            
            def delete_user(self, user_id):
                if user_id in self.users:
                    del self.users[user_id]
                return True
        
        return UserManagementService(self.db_pool)
    
    def start_services(self, services_to_start=None):
        """
        Start the specified services or all services if none specified.
        
        Args:
            services_to_start: List of service names to start
        """
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        if services_to_start is None:
            services_to_start = list(self.services.keys())
        
        self.logger.info(f"Starting services: {services_to_start}")
        
        # Start monitoring first
        self.monitoring.start()
        
        # Start each service in a separate thread
        for service_name in services_to_start:
            if service_name not in self.services:
                self.logger.warning(f"Service {service_name} not found")
                continue
            
            if service_name in self.service_threads and self.service_threads[service_name].is_alive():
                self.logger.warning(f"Service {service_name} already running")
                continue
            
            # Create and start the thread
            self._start_service_thread(service_name)
    
    def _start_service_thread(self, service_name):
        """Start a service in a separate thread."""
        service = self.services[service_name]
        
        # Define the thread target function based on service type
        if service_name == "web_interface":
            target_func = lambda: service.run(host='0.0.0.0', port=5000)
        elif service_name == "telegram_bot":
            target_func = lambda: service.run()
        elif service_name == "metaverse":
            target_func = lambda: service.run_in_background()
        else:
            # Generic background service
            target_func = lambda: self._run_background_service(service_name)
        
        # Create and start the thread
        thread = threading.Thread(
            target=target_func,
            name=f"{service_name}_thread",
            daemon=True
        )
        thread.start()
        
        # Store the thread
        self.service_threads[service_name] = thread
        self.logger.info(f"Started service: {service_name}")
    
    def _run_background_service(self, service_name):
        """Run a background service."""
        service = self.services[service_name]
        self.logger.info(f"Running background service: {service_name}")
        
        # Keep the service running
        try:
            # If service has a run method, call it
            if hasattr(service, "run"):
                service.run()
            # Otherwise just keep the thread alive
            else:
                while True:
                    time.sleep(60)
                    # Perform any periodic tasks if needed
        except Exception as e:
            self.logger.error(f"Error in service {service_name}: {e}")
    
    def stop_services(self, services_to_stop=None):
        """
        Stop the specified services or all services if none specified.
        
        Args:
            services_to_stop: List of service names to stop
        """
        if services_to_stop is None:
            services_to_stop = list(self.services.keys())
        
        self.logger.info(f"Stopping services: {services_to_stop}")
        
        # Stop each service
        for service_name in services_to_stop:
            if service_name not in self.service_threads:
                continue
            
            thread = self.service_threads[service_name]
            service = self.services[service_name]
            
            # Call stop method if it exists
            if hasattr(service, "stop"):
                service.stop()
            
            # Wait for thread to finish (with timeout)
            if thread.is_alive():
                thread.join(timeout=5)
            
            # Remove thread from dictionary
            del self.service_threads[service_name]
            
            self.logger.info(f"Stopped service: {service_name}")
        
        # Stop monitoring last
        if self.monitoring:
            self.monitoring.stop()
    
    def shutdown(self):
        """Shutdown the entire system."""
        self.logger.info("Shutting down Unified System...")
        
        # Stop all services
        self.stop_services()
        
        # Cleanup resources
        if self.db_pool:
            self.db_pool.close_all()
        
        # Shutdown improvement manager
        if hasattr(self, 'improvement_manager'):
            self.improvement_manager.shutdown()
        
        # Shutdown base system
        if hasattr(self, 'system'):
            self.system.shutdown()
        
        self.logger.info("Unified System shutdown complete")
    
    def handle_request(self, request_data, route=None):
        """
        Handle a request from any client.
        
        Args:
            request_data: Request data
            route: Optional route for API gateway
            
        Returns:
            Response data
        """
        # Validate input
        if not self.input_validator.validate_dict(request_data):
            return {"error": "Invalid request data"}
        
        # Use API gateway if route is specified
        if route:
            return self.services["api_gateway"].handle_request(route, request_data)
        
        # Otherwise, forward to assistant core
        return self.services["assistant_core"].handle_message(
            request_data.get("message", ""),
            request_data.get("user_id", "unknown"),
            request_data.get("platform", "api")
        )

def create_unified_system(config_path=None):
    """
    Create a unified system instance.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        UnifiedSystem instance
    """
    return UnifiedSystem(config_path)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified System Integrator")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--services", help="Comma-separated list of services to start")
    args = parser.parse_args()
    
    # Create unified system
    unified_system = create_unified_system(args.config)
    
    # Parse services to start
    services_to_start = None
    if args.services:
        services_to_start = args.services.split(",")
    
    try:
        # Start services
        unified_system.start_services(services_to_start)
        
        # Keep the main thread running
        print("Unified system running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Shutdown the system
        unified_system.shutdown() 