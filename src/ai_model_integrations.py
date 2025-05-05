"""
AI Model Integrations for Lucky Train

This module provides integrations for various AI models:
- OpenAI GPT Models (GPT-4, GPT-3.5-turbo, GPT-3, InstructGPT)
- Anthropic Claude Models (including Claude 3.7 Sonnet)
- Hugging Face Transformer Models (BERT, RoBERTa, DistilBERT, ALBERT)
- LLaMA and LLaMA 2
- Text and Image Generation Models
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseModelInterface(ABC):
    """Base class for all AI model interfaces."""
    
    def __init__(self, config: Dict = None):
        """Initialize the model interface.
        
        Args:
            config: Configuration for the model
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        logger.info(f"Initializing {self.name}")
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate a response based on the prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Response data
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get information about the model.
        
        Returns:
            Model information
        """
        pass

class OpenAIInterface(BaseModelInterface):
    """Interface for OpenAI models (GPT-4, GPT-3.5-turbo, GPT-3, InstructGPT)."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY") or self.config.get("api_key")
        self.model = self.config.get("model", "gpt-3.5-turbo")
        
        if not self.api_key:
            logger.warning("OpenAI API key not set - functionality will be limited")
            self.client = None
        else:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized with model: {self.model}")
            except ImportError:
                logger.error("openai package not installed")
                self.client = None
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate a response using OpenAI models.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Response data
        """
        if not self.client:
            return {"error": "OpenAI client not initialized", "text": None}
        
        try:
            system_message = kwargs.get("system_message", "You are a helpful assistant.")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 500)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "text": response.choices[0].message.content,
                "model": self.model,
                "provider": "OpenAI"
            }
        except Exception as e:
            logger.error(f"Error in OpenAI generation: {e}")
            return {"error": str(e), "text": None}
    
    def get_model_info(self) -> Dict:
        """Get information about the OpenAI model.
        
        Returns:
            Model information
        """
        models_info = {
            "gpt-4": {
                "description": "Most capable GPT-4 model, with broader general knowledge and reasoning abilities",
                "max_tokens": 8192,
                "training_data": "Up to Sep 2021"
            },
            "gpt-3.5-turbo": {
                "description": "Most capable GPT-3.5 model optimized for chat at 1/10th the cost of text-davinci-003",
                "max_tokens": 4096,
                "training_data": "Up to Sep 2021"
            },
            "gpt-3": {
                "description": "Legacy GPT-3 models with varying capabilities",
                "max_tokens": 2048,
                "training_data": "Up to Oct 2019"
            },
            "text-davinci-003": {
                "description": "InstructGPT model with improvements over vanilla GPT-3",
                "max_tokens": 4097,
                "training_data": "Up to Jun 2021"
            }
        }
        
        return models_info.get(self.model, {"description": "Unknown model", "max_tokens": "Unknown"})

class AnthropicInterface(BaseModelInterface):
    """Interface for Anthropic Claude models including Claude 3.7 Sonnet."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or self.config.get("api_key")
        self.model = self.config.get("model", "claude-3-sonnet-20240229")
        
        if not self.api_key:
            logger.warning("Anthropic API key not set - functionality will be limited")
            self.client = None
        else:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Anthropic client initialized with model: {self.model}")
            except ImportError:
                logger.error("anthropic package not installed")
                self.client = None
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate a response using Anthropic Claude models.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Response data
        """
        if not self.client:
            return {"error": "Anthropic client not initialized", "text": None}
        
        try:
            system_message = kwargs.get("system_message", "You are Claude, a helpful AI assistant.")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            response = self.client.messages.create(
                model=self.model,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "text": response.content[0].text,
                "model": self.model,
                "provider": "Anthropic"
            }
        except Exception as e:
            logger.error(f"Error in Anthropic generation: {e}")
            return {"error": str(e), "text": None}
    
    def get_model_info(self) -> Dict:
        """Get information about the Anthropic Claude model.
        
        Returns:
            Model information
        """
        models_info = {
            "claude-3-opus-20240229": {
                "description": "Most powerful Claude model for highly complex tasks",
                "max_tokens": 200000,
                "training_data": "Up to Aug 2023"
            },
            "claude-3-sonnet-20240229": {
                "description": "Balanced model with excellent performance, faster and more cost-effective",
                "max_tokens": 200000,
                "training_data": "Up to Aug 2023"
            },
            "claude-3-haiku-20240307": {
                "description": "Fastest and most compact Claude model",
                "max_tokens": 200000,
                "training_data": "Up to Aug 2023"
            },
            "claude-3.5-sonnet-20240620": {
                "description": "Claude 3.5 Sonnet model with advanced capabilities",
                "max_tokens": 200000,
                "training_data": "Up to Early 2024"
            }
        }
        
        return models_info.get(self.model, {"description": "Unknown model", "max_tokens": "Unknown"})

class HuggingFaceInterface(BaseModelInterface):
    """Interface for Hugging Face transformer models (BERT, RoBERTa, DistilBERT, ALBERT)."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = os.getenv("HF_API_KEY") or self.config.get("api_key")
        self.model_name = self.config.get("model_name", "bert-base-uncased")
        self.task = self.config.get("task", "text-classification")
        
        try:
            from transformers import AutoTokenizer, AutoModel, pipeline
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # For specific tasks
            if self.task != "embeddings":
                self.pipeline = pipeline(self.task, model=self.model_name)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            logger.info(f"HuggingFace model loaded: {self.model_name} for {self.task} on {self.device}")
            self.model_loaded = True
        except ImportError as e:
            logger.error(f"Missing dependency for HuggingFace interface: {e}")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {e}")
            self.model_loaded = False
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate a response using HuggingFace models.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Response data
        """
        if not self.model_loaded:
            return {"error": "HuggingFace model not loaded", "text": None}
        
        try:
            # Handle different tasks
            if self.task == "text-generation":
                max_length = kwargs.get("max_length", 50)
                result = self.pipeline(prompt, max_length=max_length)
                return {
                    "text": result[0]["generated_text"],
                    "model": self.model_name,
                    "provider": "HuggingFace"
                }
            
            elif self.task == "text-classification":
                result = self.pipeline(prompt)
                return {
                    "label": result[0]["label"],
                    "score": result[0]["score"],
                    "model": self.model_name,
                    "provider": "HuggingFace"
                }
            
            elif self.task == "embeddings":
                import torch
                
                # Tokenize and get model output
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Use the [CLS] token embedding as the sentence embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                return {
                    "embeddings": embeddings.tolist()[0],
                    "dimensions": len(embeddings.tolist()[0]),
                    "model": self.model_name,
                    "provider": "HuggingFace"
                }
            
            else:
                return {"error": f"Unsupported task: {self.task}", "text": None}
                
        except Exception as e:
            logger.error(f"Error in HuggingFace generation: {e}")
            return {"error": str(e), "text": None}
    
    def get_model_info(self) -> Dict:
        """Get information about the HuggingFace model.
        
        Returns:
            Model information
        """
        model_families = {
            "bert": {
                "description": "Bidirectional Encoder Representations from Transformers",
                "paper": "https://arxiv.org/abs/1810.04805",
                "variants": ["bert-base-uncased", "bert-large-uncased", "bert-base-cased"]
            },
            "roberta": {
                "description": "A Robustly Optimized BERT Pretraining Approach",
                "paper": "https://arxiv.org/abs/1907.11692",
                "variants": ["roberta-base", "roberta-large"]
            },
            "distilbert": {
                "description": "Distilled version of BERT, smaller and faster with similar performance",
                "paper": "https://arxiv.org/abs/1910.01108",
                "variants": ["distilbert-base-uncased", "distilbert-base-cased"]
            },
            "albert": {
                "description": "A Lite BERT with parameter reduction techniques",
                "paper": "https://arxiv.org/abs/1909.11942",
                "variants": ["albert-base-v2", "albert-large-v2", "albert-xlarge-v2"]
            }
        }
        
        # Determine model family
        family = next((k for k in model_families.keys() if k in self.model_name.lower()), None)
        
        if family:
            info = model_families[family].copy()
            info["model_name"] = self.model_name
            info["task"] = self.task
            return info
        else:
            return {
                "model_name": self.model_name,
                "task": self.task,
                "description": "Custom transformer model"
            }

class LlamaInterface(BaseModelInterface):
    """Interface for LLaMA and LLaMA 2 models."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model_path = self.config.get("model_path", "models/llama-2-7b-chat.gguf")
        self.context_size = self.config.get("context_size", 2048)
        self.threads = self.config.get("threads", 4)
        
        try:
            from llama_cpp import Llama
            
            if os.path.exists(self.model_path):
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_size,
                    n_threads=self.threads
                )
                logger.info(f"LLaMA model loaded from {self.model_path}")
                self.model_loaded = True
            else:
                logger.error(f"LLaMA model file not found at {self.model_path}")
                self.model_loaded = False
                
        except ImportError:
            logger.error("llama-cpp-python package not installed")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Error loading LLaMA model: {e}")
            self.model_loaded = False
    
    def generate(self, prompt: str, **kwargs) -> Dict:
        """Generate a response using LLaMA models.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Response data
        """
        if not self.model_loaded:
            return {"error": "LLaMA model not loaded", "text": None}
        
        try:
            max_tokens = kwargs.get("max_tokens", 512)
            temperature = kwargs.get("temperature", 0.8)
            top_p = kwargs.get("top_p", 0.95)
            
            # Format prompt for chat
            if "llama-2" in self.model_path.lower() and kwargs.get("chat_format", True):
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt
            
            response = self.llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            return {
                "text": response["choices"][0]["text"],
                "model": os.path.basename(self.model_path),
                "provider": "LLaMA"
            }
        except Exception as e:
            logger.error(f"Error in LLaMA generation: {e}")
            return {"error": str(e), "text": None}
    
    def get_model_info(self) -> Dict:
        """Get information about the LLaMA model.
        
        Returns:
            Model information
        """
        model_name = os.path.basename(self.model_path)
        
        llama_info = {
            "llama-7b": {
                "description": "Original LLaMA 7 billion parameter model",
                "parameters": "7B",
                "context_length": 2048,
                "training_data": "Up to early 2023"
            },
            "llama-13b": {
                "description": "Original LLaMA 13 billion parameter model",
                "parameters": "13B",
                "context_length": 2048,
                "training_data": "Up to early 2023"
            },
            "llama-2-7b": {
                "description": "LLaMA 2 7 billion parameter base model",
                "parameters": "7B",
                "context_length": 4096,
                "training_data": "Up to early 2023"
            },
            "llama-2-7b-chat": {
                "description": "LLaMA 2 7 billion parameter fine-tuned for chat",
                "parameters": "7B",
                "context_length": 4096,
                "training_data": "Up to early 2023"
            },
            "llama-2-13b": {
                "description": "LLaMA 2 13 billion parameter base model",
                "parameters": "13B",
                "context_length": 4096,
                "training_data": "Up to early 2023"
            },
            "llama-2-13b-chat": {
                "description": "LLaMA 2 13 billion parameter fine-tuned for chat",
                "parameters": "13B",
                "context_length": 4096,
                "training_data": "Up to early 2023"
            },
            "llama-2-70b": {
                "description": "LLaMA 2 70 billion parameter base model",
                "parameters": "70B",
                "context_length": 4096,
                "training_data": "Up to early 2023"
            },
            "llama-2-70b-chat": {
                "description": "LLaMA 2 70 billion parameter fine-tuned for chat",
                "parameters": "70B",
                "context_length": 4096,
                "training_data": "Up to early 2023"
            }
        }
        
        # Try to match the model name with known models
        for key, info in llama_info.items():
            if key in model_name.lower():
                result = info.copy()
                result["model_name"] = model_name
                return result
        
        # Default info if not found
        return {
            "model_name": model_name,
            "description": "Custom LLaMA model variant",
            "context_length": self.context_size
        }

# Utility function to create model interfaces
def create_model_interface(model_type: str, config: Dict = None) -> BaseModelInterface:
    """Create and return a model interface based on the model type.
    
    Args:
        model_type: Type of model interface to create
        config: Configuration for the model
        
    Returns:
        Model interface instance
    """
    model_interfaces = {
        "openai": OpenAIInterface,
        "gpt": OpenAIInterface,
        "anthropic": AnthropicInterface,
        "claude": AnthropicInterface,
        "huggingface": HuggingFaceInterface,
        "transformers": HuggingFaceInterface,
        "llama": LlamaInterface
    }
    
    interface_class = model_interfaces.get(model_type.lower())
    
    if interface_class:
        return interface_class(config)
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}") 