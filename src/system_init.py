"""
System Initialization Module for Lucky Train AI Assistant

This module initializes and manages all the system components:
- Security (authentication, authorization)
- Internal database
- Caching and question caching
- Logging
- Vector database and knowledge base
"""

import os
import logging
import json
from typing import Dict, Any, Optional

# Import new components
from security import SecurityManager, RBAC
from internal_db import get_db, InternalDatabase
from question_cache import get_question_cache, QuestionCache
from caching import CacheManager
from logger import get_logger, LuckyTrainLogger
from vector_db import VectorDBHandler

# Configuration file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')

class SystemManager:
    """Main system manager that initializes and provides access to all components."""
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, config: Dict = None):
        """Initialize the system manager.
        
        Args:
            config_path: Path to configuration file
            config: Configuration dictionary (overrides file if provided)
        """
        # Load configuration
        self.config = config if config is not None else self._load_config(config_path)
        
        # Initialize logger first so other components can use it
        logger_config = self.config.get('logger_settings', {})
        self.logger = get_logger('lucky_train', logger_config)
        
        self.logger.info("Initializing Lucky Train AI system")
        
        # Initialize security components
        security_config = self.config.get('security_settings', {})
        self.security_manager = SecurityManager(security_config)
        self.rbac = RBAC(security_config.get('rbac', {}))
        
        # Initialize database
        db_config = self.config.get('database_settings', {})
        self.db = get_db(db_config)
        
        # Initialize caching
        cache_config = self.config.get('cache_settings', {})
        self.cache_manager = CacheManager(cache_config)
        
        # Initialize question cache
        question_cache_config = self.config.get('question_cache_settings', {})
        self.question_cache = get_question_cache(question_cache_config)
        
        # Initialize vector database handler for knowledge base
        vector_db_config = self.config.get('vector_db_settings', {})
        try:
            self.vector_db = VectorDBHandler(config_path)
            self.logger.info("Vector database handler initialized")
            self._load_knowledge_base()
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            self.vector_db = None
        
        self.logger.info("Lucky Train AI system initialization complete")
    
    def _load_knowledge_base(self):
        """Load the JSON knowledge base files into the vector database."""
        if not self.vector_db:
            self.logger.warning("Vector database not initialized, skipping knowledge base loading")
            return
            
        try:
            # Knowledge base directory
            kb_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge_base')
            
            if not os.path.exists(kb_dir):
                self.logger.warning(f"Knowledge base directory {kb_dir} not found")
                # Create the directory if it doesn't exist
                try:
                    os.makedirs(kb_dir, exist_ok=True)
                    self.logger.info(f"Created knowledge base directory: {kb_dir}")
                except Exception as e:
                    self.logger.error(f"Failed to create knowledge base directory: {e}")
                return
                
            # Check if there are any JSON files in the directory
            json_files = [f for f in os.listdir(kb_dir) if f.endswith('.json')]
            if not json_files:
                self.logger.warning(f"No JSON files found in knowledge base directory {kb_dir}")
                return
                
            # Load all JSON files in the knowledge base directory
            for filename in json_files:
                try:
                    filepath = os.path.join(kb_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        kb_data = json.load(f)
                        
                    # Load the knowledge base into the vector database
                    if hasattr(self.vector_db, 'load_knowledge_base'):
                        self.vector_db.load_knowledge_base({
                            "name": filename.replace('.json', ''),
                            "data": kb_data
                        })
                        self.logger.info(f"Loaded knowledge base from {filename}")
                    else:
                        self.logger.error(f"Vector database does not support 'load_knowledge_base' method")
                        break
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing JSON from {filename}: {e}")
                except Exception as e:
                    self.logger.error(f"Error loading knowledge base from {filename}: {e}")
            
            self.logger.info("Knowledge base loading complete")
        except Exception as e:
            self.logger.error(f"Error in knowledge base loading: {e}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # If config file doesn't exist, create default
                return self._create_default_config(config_path)
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return self._create_default_config(config_path)
    
    def _create_default_config(self, config_path: str) -> Dict:
        """Create and save a default configuration.
        
        Args:
            config_path: Path to save the configuration
            
        Returns:
            Default configuration dictionary
        """
        # Create default configuration
        default_config = {
            'security_settings': {
                'jwt_expiry_hours': 24,
                'rate_limit': 60,
                'rate_limits': {
                    'login': 10,
                    'register': 5,
                    'api': 100
                },
                'rbac': {
                    'roles': {
                        'admin': {
                            'description': 'Administrator with full access',
                            'permissions': ['*']
                        },
                        'moderator': {
                            'description': 'Moderator with content management access',
                            'permissions': ['read:*', 'write:content', 'edit:content', 'delete:content']
                        },
                        'premium': {
                            'description': 'Premium user with enhanced access',
                            'permissions': ['read:*', 'write:own', 'edit:own']
                        },
                        'user': {
                            'description': 'Standard user',
                            'permissions': ['read:public', 'write:own', 'edit:own']
                        },
                        'guest': {
                            'description': 'Guest user with limited access',
                            'permissions': ['read:public']
                        }
                    },
                    'role_hierarchy': {
                        'admin': 100,
                        'moderator': 50,
                        'premium': 20,
                        'user': 10,
                        'guest': 1
                    }
                }
            },
            'database_settings': {
                'db_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'lucky_train.db')
            },
            'cache_settings': {
                'enabled': True,
                'type': 'memory',
                'ttl_seconds': 3600,
                'max_size': 1000
            },
            'question_cache_settings': {
                'enabled': True,
                'similarity_threshold': 0.85,
                'ttl_seconds': 86400,
                'max_size': 10000,
                'model_name': 'paraphrase-MiniLM-L6-v2',
                'persist_file': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'question_cache.pkl'),
                'persist_on_shutdown': True,
                'load_on_startup': True,
                'user_specific': False,
                'context_aware': True
            },
            'logger_settings': {
                'log_level': 'INFO',
                'log_file': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'lucky_train_ai.log'),
                'error_log_file': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'error.log'),
                'access_log_file': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'access.log'),
                'security_log_file': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'security.log'),
                'max_bytes': 10 * 1024 * 1024,  # 10 MB
                'backup_count': 5,
                'use_json': False,
                'async_logging': True
            },
            'vector_db_settings': {
                'db_type': 'qdrant',  # Options: qdrant, pinecone, weaviate
                'qdrant_url': os.getenv('QDRANT_URL'),
                'qdrant_api_key': os.getenv('QDRANT_API_KEY'),
                'qdrant_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'qdrant_data'),
                'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
                'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT'),
                'pinecone_index': 'lucky-train',
                'weaviate_url': os.getenv('WEAVIATE_URL'),
                'weaviate_api_key': os.getenv('WEAVIATE_API_KEY'),
                'collection_name': 'lucky_train_kb',
                'embedding_model': {
                    'type': 'openai',  # Options: openai, sentence_transformers
                    'name': 'text-embedding-3-small'  # For sentence_transformers: 'all-MiniLM-L6-v2'
                }
            }
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save default configuration
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            logging.info(f"Created default configuration at {config_path}")
        except Exception as e:
            logging.error(f"Error creating default configuration: {e}")
        
        return default_config
    
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self.logger.info("Shutting down Lucky Train AI system")
        
        # Save question cache if configured
        if hasattr(self, 'question_cache') and self.question_cache:
            config = self.config.get('question_cache_settings', {})
            if config.get('persist_on_shutdown', True) and config.get('persist_file'):
                self.question_cache.save()
        
        self.logger.info("Lucky Train AI system shutdown complete")

# Singleton system manager instance
_system_manager = None

def init_system(config_path: str = DEFAULT_CONFIG_PATH, config: Dict = None) -> SystemManager:
    """Initialize the system.
    
    Args:
        config_path: Path to configuration file
        config: Configuration dictionary (overrides file if provided)
        
    Returns:
        System manager instance
    """
    global _system_manager
    
    if _system_manager is None:
        _system_manager = SystemManager(config_path, config)
    
    return _system_manager

def get_system() -> Optional[SystemManager]:
    """Get the system manager instance.
    
    Returns:
        System manager instance or None if not initialized
    """
    return _system_manager 