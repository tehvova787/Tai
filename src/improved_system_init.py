"""
Improved System Initialization Module for Lucky Train AI Assistant

This module initializes and manages all the system components with enhanced security, 
performance, and reliability features.
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional

# Import original system components
from system_init import SystemManager as OriginalSystemManager

# Import improvement system
from system_improvements import enhance_system

# Import improved modules
from security_improvements import SecretManager
from memory_monitor import get_memory_monitor
from logger_enhancements import get_logger
from db_connection_pool import get_connection_pool

# Configure basic logging until enhanced logging is set up
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration file paths
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')
IMPROVEMENTS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'improvements.json')

class ImprovedSystemManager(OriginalSystemManager):
    """Enhanced system manager with improvements for security, memory, etc."""
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, config: Dict = None):
        """Initialize the improved system manager.
        
        Args:
            config_path: Path to configuration file
            config: Configuration dictionary (overrides file if provided)
        """
        # Initialize base system
        super().__init__(config_path, config)
        
        # Initialize improvements
        self.improvement_manager = enhance_system(self, IMPROVEMENTS_CONFIG_PATH)
        
        # Log initialization complete with improvements
        if hasattr(self, 'logger'):
            self.logger.info("Lucky Train AI system initialization complete with improvements")
        else:
            logger.info("Lucky Train AI system initialization complete with improvements")
    
    def shutdown(self) -> None:
        """Perform enhanced cleanup and shutdown operations."""
        logger.info("Shutting down improved Lucky Train AI system")
        
        # Shutdown improvements first
        if hasattr(self, 'improvement_manager'):
            self.improvement_manager.shutdown()
        
        # Call original shutdown
        super().shutdown()
        
        logger.info("Improved Lucky Train AI system shutdown complete")

# Enhanced singleton system manager instance
_improved_system_manager = None

def init_system(config_path: str = DEFAULT_CONFIG_PATH, config: Dict = None) -> ImprovedSystemManager:
    """Initialize the improved system.
    
    Args:
        config_path: Path to configuration file
        config: Configuration dictionary (overrides file if provided)
        
    Returns:
        Improved system manager instance
    """
    global _improved_system_manager
    
    if _improved_system_manager is None:
        _improved_system_manager = ImprovedSystemManager(config_path, config)
    
    return _improved_system_manager

def get_system() -> Optional[ImprovedSystemManager]:
    """Get the improved system manager instance.
    
    Returns:
        Improved system manager instance or None if not initialized
    """
    return _improved_system_manager

# Example usage
if __name__ == "__main__":
    # Initialize the system with improvements
    system = init_system()
    
    # Use the system...
    
    # Graceful shutdown
    system.shutdown() 