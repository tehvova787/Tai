"""
Unified AI Interface for Lucky Train AI

This module provides a unified interface for different AI models,
including both cloud-based and local models.
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Generator
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIModel(ABC):
    """Abstract base class for all AI models."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response to a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate a streaming response to a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the generated response
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the model name.
        
        Returns:
            Model name
        """
        pass
    
    @property
    @abstractmethod
    def type(self) -> str:
        """Get the model type.
        
        Returns:
            Model type (cloud, local, hybrid)
        """
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, bool]:
        """Get the model capabilities.
        
        Returns:
            Dictionary of capability flags
        """
        pass

class OpenAIModel(AIModel):
    """OpenAI API-based model."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        """Initialize the OpenAI model.
        
        Args:
            model_name: Model name
            api_key: OpenAI API key (defaults to environment variable)
        """
        import openai
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Determine capabilities based on model name
        self.model_capabilities = {
            "chat": True,
            "streaming": True,
            "embeddings": "text-embedding" in model_name,
            "function_calling": "gpt-4" in model_name or "gpt-3.5" in model_name,
            "image_understanding": "vision" in model_name or "gpt-4" in model_name,
            "long_context": "16k" in model_name or "32k" in model_name
        }
        
        logger.info(f"Initialized OpenAI model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the OpenAI API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters including:
                - system_prompt: System prompt
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - functions: Functions for function calling
                
        Returns:
            Generated response
        """
        try:
            # Extract parameters
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            functions = kwargs.get("functions", None)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Prepare completion parameters
            completion_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add functions if provided
            if functions and self.model_capabilities["function_calling"]:
                completion_params["functions"] = functions
            
            # Make the API call
            response = self.client.chat.completions.create(**completion_params)
            
            # Extract the response text
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {e}")
            return f"Error: {str(e)}"
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate a streaming response using the OpenAI API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters including:
                - system_prompt: System prompt
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                
        Yields:
            Chunks of the generated response
        """
        try:
            # Extract parameters
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Make the API call with streaming enabled
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Stream the response
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"Error streaming response from OpenAI: {e}")
            yield f"Error: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using the OpenAI API.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        try:
            # Choose an appropriate embedding model
            embedding_model = "text-embedding-3-small"
            
            # Make the API call
            response = self.client.embeddings.create(
                input=texts,
                model=embedding_model
            )
            
            # Extract and return the embeddings
            return [data.embedding for data in response.data]
        
        except Exception as e:
            logger.error(f"Error getting embeddings from OpenAI: {e}")
            # Return empty embeddings
            return [[0.0] * 1536] * len(texts)
    
    @property
    def name(self) -> str:
        """Get the model name.
        
        Returns:
            Model name
        """
        return self.model_name
    
    @property
    def type(self) -> str:
        """Get the model type.
        
        Returns:
            Model type
        """
        return "cloud"
    
    @property
    def capabilities(self) -> Dict[str, bool]:
        """Get the model capabilities.
        
        Returns:
            Dictionary of capability flags
        """
        return self.model_capabilities

class LocalLLMModel(AIModel):
    """Local LLM model using llama.cpp, GGML, or similar backends."""
    
    def __init__(self, model_path: str, model_type: str = "llama", **kwargs):
        """Initialize the local LLM model.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (llama, mistral, etc.)
            **kwargs: Additional parameters including:
                - context_size: Context window size
                - threads: Number of CPU threads to use
        """
        self.model_path = model_path
        self.model_type = model_type
        self.context_size = kwargs.get("context_size", 4096)
        self.threads = kwargs.get("threads", 4)
        
        # Initialize the model
        try:
            from llama_cpp import Llama
            
            self.llm = Llama(
                model_path=model_path,
                n_ctx=self.context_size,
                n_threads=self.threads
            )
            
            # Determine model name from path
            import os
            self.model_name = os.path.basename(model_path)
            
            # Set capabilities based on model type
            self.model_capabilities = {
                "chat": True,
                "streaming": True,
                "embeddings": False,  # Most local models don't handle embeddings well
                "function_calling": False,  # Function calling typically not supported
                "image_understanding": False,
                "long_context": self.context_size > 8192
            }
            
            logger.info(f"Initialized local LLM: {self.model_name}")
        
        except ImportError:
            raise ImportError("llama-cpp-python package not installed. Run 'pip install llama-cpp-python'")
        except Exception as e:
            raise ValueError(f"Error initializing local LLM: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the local LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters including:
                - system_prompt: System prompt
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                
        Returns:
            Generated response
        """
        try:
            # Extract parameters
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            # Prepare the full prompt
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            # Generate the response
            output = self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["User:", "\n\nUser"],
                echo=False
            )
            
            # Extract and return the response text
            return output["choices"][0]["text"].strip()
        
        except Exception as e:
            logger.error(f"Error generating response from local LLM: {e}")
            return f"Error: {str(e)}"
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate a streaming response using the local LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters including:
                - system_prompt: System prompt
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                
        Yields:
            Chunks of the generated response
        """
        try:
            # Extract parameters
            system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            # Prepare the full prompt
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            # Generate the streaming response
            for output in self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["User:", "\n\nUser"],
                echo=False,
                stream=True
            ):
                if output and output["choices"] and output["choices"][0]["text"]:
                    yield output["choices"][0]["text"]
        
        except Exception as e:
            logger.error(f"Error streaming response from local LLM: {e}")
            yield f"Error: {str(e)}"
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using a local embedding model.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        try:
            # For local models, we typically use a separate embedding model
            # from sentence-transformers
            from sentence_transformers import SentenceTransformer
            
            # Load the model (lazy loading - only happens once)
            if not hasattr(self, 'embedding_model'):
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
            
            # Convert to list format
            return embeddings.tolist()
        
        except ImportError:
            logger.error("sentence-transformers package not installed. Run 'pip install sentence-transformers'")
            # Return empty embeddings with the correct size
            return [[0.0] * 384] * len(texts)
        
        except Exception as e:
            logger.error(f"Error getting embeddings from local model: {e}")
            # Return empty embeddings with the correct size
            return [[0.0] * 384] * len(texts)
    
    @property
    def name(self) -> str:
        """Get the model name.
        
        Returns:
            Model name
        """
        return self.model_name
    
    @property
    def type(self) -> str:
        """Get the model type.
        
        Returns:
            Model type
        """
        return "local"
    
    @property
    def capabilities(self) -> Dict[str, bool]:
        """Get the model capabilities.
        
        Returns:
            Dictionary of capability flags
        """
        return self.model_capabilities

class HybridModel(AIModel):
    """Hybrid model that combines cloud and local models based on requirements."""
    
    def __init__(self, primary_model: AIModel, backup_model: AIModel):
        """Initialize the hybrid model.
        
        Args:
            primary_model: Primary model (usually cloud-based)
            backup_model: Backup model (usually local)
        """
        self.primary_model = primary_model
        self.backup_model = backup_model
        self.model_name = f"Hybrid({primary_model.name}+{backup_model.name})"
        
        # Merge capabilities, prioritizing the primary model
        self.model_capabilities = {
            capability: primary_model.capabilities.get(capability, False) or backup_model.capabilities.get(capability, False)
            for capability in set(primary_model.capabilities) | set(backup_model.capabilities)
        }
        
        logger.info(f"Initialized hybrid model: {self.model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the appropriate model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        try:
            # Try primary model first
            return self.primary_model.generate(prompt, **kwargs)
        except Exception as e:
            logger.warning(f"Primary model failed, falling back to backup: {e}")
            return self.backup_model.generate(prompt, **kwargs)
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generate a streaming response using the appropriate model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the generated response
        """
        try:
            # Try streaming from primary model
            for chunk in self.primary_model.generate_stream(prompt, **kwargs):
                yield chunk
        except Exception as e:
            logger.warning(f"Primary model streaming failed, falling back to backup: {e}")
            yield "Primary model unavailable, switching to backup:\n"
            for chunk in self.backup_model.generate_stream(prompt, **kwargs):
                yield chunk
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using the appropriate model.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        try:
            # Try primary model first for embeddings
            return self.primary_model.get_embeddings(texts)
        except Exception as e:
            logger.warning(f"Primary model embeddings failed, falling back to backup: {e}")
            return self.backup_model.get_embeddings(texts)
    
    @property
    def name(self) -> str:
        """Get the model name.
        
        Returns:
            Model name
        """
        return self.model_name
    
    @property
    def type(self) -> str:
        """Get the model type.
        
        Returns:
            Model type
        """
        return "hybrid"
    
    @property
    def capabilities(self) -> Dict[str, bool]:
        """Get the model capabilities.
        
        Returns:
            Dictionary of capability flags
        """
        return self.model_capabilities

class ModelManager:
    """Manager for different AI models."""
    
    def __init__(self, config: Dict = None):
        """Initialize the model manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.default_model = None
        self.lock = threading.RLock()
        
        # Initialize models from config
        self._initialize_models()
        
        logger.info(f"Initialized model manager with {len(self.models)} models")
    
    def _initialize_models(self) -> None:
        """Initialize models from configuration."""
        with self.lock:
            # Get model configurations
            model_configs = self.config.get("models", {})
            
            for model_id, model_config in model_configs.items():
                if not model_config.get("enabled", True):
                    continue
                
                model_type = model_config.get("type", "cloud")
                
                try:
                    if model_type == "cloud":
                        model = self._init_cloud_model(model_id, model_config)
                    elif model_type == "local":
                        model = self._init_local_model(model_id, model_config)
                    elif model_type == "hybrid":
                        model = self._init_hybrid_model(model_id, model_config)
                    else:
                        logger.warning(f"Unknown model type: {model_type}")
                        continue
                    
                    self.models[model_id] = model
                    
                    # Set as default if specified
                    if model_config.get("default", False):
                        self.default_model = model_id
                
                except Exception as e:
                    logger.error(f"Error initializing model {model_id}: {e}")
            
            # If no default model is set, use the first one
            if not self.default_model and self.models:
                self.default_model = next(iter(self.models))
    
    def _init_cloud_model(self, model_id: str, model_config: Dict) -> AIModel:
        """Initialize a cloud-based model.
        
        Args:
            model_id: Model ID
            model_config: Model configuration
            
        Returns:
            Initialized AI model
        """
        provider = model_config.get("provider", "openai")
        
        if provider == "openai":
            return OpenAIModel(
                model_name=model_config.get("model_name", "gpt-3.5-turbo"),
                api_key=model_config.get("api_key")
            )
        
        raise ValueError(f"Unsupported cloud provider: {provider}")
    
    def _init_local_model(self, model_id: str, model_config: Dict) -> AIModel:
        """Initialize a local model.
        
        Args:
            model_id: Model ID
            model_config: Model configuration
            
        Returns:
            Initialized AI model
        """
        model_path = model_config.get("model_path")
        
        if not model_path:
            raise ValueError("Model path not specified for local model")
        
        return LocalLLMModel(
            model_path=model_path,
            model_type=model_config.get("model_type", "llama"),
            context_size=model_config.get("context_size", 4096),
            threads=model_config.get("threads", 4)
        )
    
    def _init_hybrid_model(self, model_id: str, model_config: Dict) -> AIModel:
        """Initialize a hybrid model.
        
        Args:
            model_id: Model ID
            model_config: Model configuration
            
        Returns:
            Initialized AI model
        """
        primary_id = model_config.get("primary_model")
        backup_id = model_config.get("backup_model")
        
        if not primary_id or not backup_id:
            raise ValueError("Primary and backup models must be specified for hybrid model")
        
        # Initialize primary and backup models if not already initialized
        if primary_id not in self.models:
            primary_config = self.config.get("models", {}).get(primary_id, {})
            
            if not primary_config:
                raise ValueError(f"Configuration for primary model {primary_id} not found")
            
            primary_type = primary_config.get("type", "cloud")
            
            if primary_type == "cloud":
                primary_model = self._init_cloud_model(primary_id, primary_config)
            elif primary_type == "local":
                primary_model = self._init_local_model(primary_id, primary_config)
            else:
                raise ValueError(f"Unsupported model type for primary model: {primary_type}")
            
            self.models[primary_id] = primary_model
        
        if backup_id not in self.models:
            backup_config = self.config.get("models", {}).get(backup_id, {})
            
            if not backup_config:
                raise ValueError(f"Configuration for backup model {backup_id} not found")
            
            backup_type = backup_config.get("type", "local")
            
            if backup_type == "cloud":
                backup_model = self._init_cloud_model(backup_id, backup_config)
            elif backup_type == "local":
                backup_model = self._init_local_model(backup_id, backup_config)
            else:
                raise ValueError(f"Unsupported model type for backup model: {backup_type}")
            
            self.models[backup_id] = backup_model
        
        return HybridModel(
            primary_model=self.models[primary_id],
            backup_model=self.models[backup_id]
        )
    
    def get_model(self, model_id: str = None) -> AIModel:
        """Get a model by ID.
        
        Args:
            model_id: Model ID (uses default if not specified)
            
        Returns:
            AI model
        """
        with self.lock:
            model_id = model_id or self.default_model
            
            if not model_id:
                raise ValueError("No model ID specified and no default model available")
            
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            return self.models[model_id]
    
    def list_models(self) -> Dict[str, Dict]:
        """List all available models.
        
        Returns:
            Dictionary of model information
        """
        with self.lock:
            return {
                model_id: {
                    "name": model.name,
                    "type": model.type,
                    "capabilities": model.capabilities,
                    "is_default": model_id == self.default_model
                }
                for model_id, model in self.models.items()
            }

# Singleton instance
_model_manager = None

def get_model_manager(config: Dict = None) -> ModelManager:
    """Get the model manager instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model manager instance
    """
    global _model_manager
    
    if _model_manager is None:
        _model_manager = ModelManager(config)
    
    return _model_manager 