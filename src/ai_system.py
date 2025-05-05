"""
AI System for Lucky Train - Integration of AI models and database connectors

This module brings together the various AI model types and database connectors
to create a unified AI system for the Lucky Train assistant.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv

# Import AI models and database connectors
from ai_models import create_ai_model, BaseAIModel
from database_connectors import create_db_connector, BaseDBConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AISystem:
    """Unified AI system combining multiple AI models and database connectors."""
    
    def __init__(self, config_path: str = "../config/config.json"):
        """Initialize the AI system.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.ai_models = {}
        self.db_connectors = {}
        
        # Initialize AI models
        self._initialize_ai_models()
        
        # Initialize database connectors
        self._initialize_db_connectors()
        
        # Set default AI model
        self.current_model_type = self.config.get("default_ai_model", "ani")
        
        logger.info(f"AI System initialized with {len(self.ai_models)} models and {len(self.db_connectors)} database connectors")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load the configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            The configuration as a dictionary
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            # Return default configuration
            return {
                "default_ai_model": "ani",
                "enabled_ai_models": ["ani", "agi", "machine_learning"],
                "enabled_db_connectors": ["chat2db"]
            }
    
    def _initialize_ai_models(self):
        """Initialize AI models based on configuration."""
        enabled_models = self.config.get("enabled_ai_models", [])
        model_configs = self.config.get("ai_model_configs", {})
        
        # If no models specified, enable a default set
        if not enabled_models:
            enabled_models = ["ani", "agi", "machine_learning"]
        
        for model_type in enabled_models:
            model_config = model_configs.get(model_type, {})
            try:
                model = create_ai_model(model_type, model_config)
                if model:
                    self.ai_models[model_type] = model
                    logger.info(f"Initialized AI model: {model_type}")
            except Exception as e:
                logger.error(f"Failed to initialize AI model {model_type}: {e}")
    
    def _initialize_db_connectors(self):
        """Initialize database connectors based on configuration."""
        enabled_connectors = self.config.get("enabled_db_connectors", [])
        connector_configs = self.config.get("db_connector_configs", {})
        
        # If no connectors specified, enable a default set if credentials are available
        if not enabled_connectors:
            # Check for available credentials in environment variables
            if os.getenv("CHAT2DB_API_ENDPOINT"):
                enabled_connectors.append("chat2db")
            if os.getenv("BIGQUERY_PROJECT_ID"):
                enabled_connectors.append("bigquery")
            if os.getenv("AURORA_HOST"):
                enabled_connectors.append("aurora")
            if os.getenv("COSMOS_ENDPOINT"):
                enabled_connectors.append("cosmosdb")
            if os.getenv("SNOWFLAKE_ACCOUNT"):
                enabled_connectors.append("snowflake")
        
        for connector_type in enabled_connectors:
            connector_config = connector_configs.get(connector_type, {})
            try:
                connector = create_db_connector(connector_type, connector_config)
                if connector:
                    self.db_connectors[connector_type] = connector
                    logger.info(f"Initialized database connector: {connector_type}")
            except Exception as e:
                logger.error(f"Failed to initialize database connector {connector_type}: {e}")
    
    def set_ai_model(self, model_type: str) -> bool:
        """Set the current AI model to use.
        
        Args:
            model_type: Type of AI model to use
            
        Returns:
            True if successful, False otherwise
        """
        if model_type in self.ai_models:
            self.current_model_type = model_type
            logger.info(f"Set current AI model to: {model_type}")
            return True
        else:
            logger.warning(f"AI model {model_type} not available")
            return False
    
    def get_current_model(self) -> BaseAIModel:
        """Get the current AI model instance.
        
        Returns:
            Current AI model
        """
        return self.ai_models.get(self.current_model_type)
    
    def get_available_models(self) -> List[str]:
        """Get a list of available AI models.
        
        Returns:
            List of available AI model types
        """
        return list(self.ai_models.keys())
    
    def get_available_connectors(self) -> List[str]:
        """Get a list of available database connectors.
        
        Returns:
            List of available database connector types
        """
        return list(self.db_connectors.keys())
    
    def connect_to_database(self, db_type: str) -> bool:
        """Connect to a specific database.
        
        Args:
            db_type: Type of database to connect to
            
        Returns:
            True if connected successfully, False otherwise
        """
        if db_type not in self.db_connectors:
            logger.warning(f"Database connector {db_type} not available")
            return False
        
        return self.db_connectors[db_type].connect()
    
    def disconnect_from_database(self, db_type: str) -> bool:
        """Disconnect from a specific database.
        
        Args:
            db_type: Type of database to disconnect from
            
        Returns:
            True if disconnected successfully, False otherwise
        """
        if db_type not in self.db_connectors:
            logger.warning(f"Database connector {db_type} not available")
            return False
        
        return self.db_connectors[db_type].disconnect()
    
    def execute_database_query(self, db_type: str, query: str, params: Dict = None) -> Dict:
        """Execute a query on a specific database.
        
        Args:
            db_type: Type of database to query
            query: Query string
            params: Query parameters
            
        Returns:
            Query results
        """
        if db_type not in self.db_connectors:
            error_msg = f"Database connector {db_type} not available"
            logger.warning(error_msg)
            return {"error": error_msg, "success": False}
        
        return self.db_connectors[db_type].execute_query(query, params)
    
    def generate_response(self, query: str, context: List[Dict] = None, model_type: str = None, **kwargs) -> Dict:
        """Generate a response using the specified or current AI model.
        
        Args:
            query: User's query
            context: Contextual information
            model_type: Type of AI model to use (if None, use current)
            
        Returns:
            Response data
        """
        # Use specified model or current model
        model_type = model_type or self.current_model_type
        
        if model_type not in self.ai_models:
            error_msg = f"AI model {model_type} not available"
            logger.warning(error_msg)
            return {"response": error_msg, "success": False}
        
        model = self.ai_models[model_type]
        
        try:
            # Log the request
            logger.info(f"Generating response with model: {model_type}, query: {query[:100]}...")
            
            # Generate response
            result = model.generate_response(query, context, **kwargs)
            
            # Add success flag if not present
            if "success" not in result:
                result["success"] = True
            
            return result
        except Exception as e:
            error_msg = f"Error generating response with {model_type}: {e}"
            logger.error(error_msg)
            return {"response": "I encountered an error processing your request.", "success": False} 