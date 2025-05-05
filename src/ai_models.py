#!/usr/bin/env python3
"""
AI Models for the Lucky Train AI Assistant

This module contains different AI model implementations used by the Lucky Train AI Assistant.
All models consistently use environment variables for API keys and credentials.
"""

import os
import time
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

try:
    import openai
except ImportError:
    openai = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAIModel(ABC):
    """Base abstract class for all AI models."""
    
    def __init__(self, config: Dict = None):
        """Initialize the AI model with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def generate_response(self, query: str, context: List[Dict] = None, **kwargs) -> Dict:
        """Generate a response for the given query.
        
        Args:
            query: User's query
            context: Retrieved knowledge contexts
            
        Returns:
            Response data
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get the list of capabilities for this AI model.
        
        Returns:
            List of capability strings
        """
        pass

class OpenAIModel(BaseAIModel):
    """OpenAI API-based model implementation."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model = self.config.get("model", "gpt-4o-mini")
        self.capabilities = ["natural_language_processing", "question_answering", 
                           "context_understanding", "knowledge_integration"]
        
        if self.openai_api_key:
            self.client = openai.OpenAI(
                api_key=self.openai_api_key,
                organization=os.getenv("OPENAI_ORGANIZATION_ID")
            )
        else:
            logger.warning("OpenAI API key not set - model will have limited functionality")
            self.client = None
    
    def generate_response(self, query: str, context: List[Dict] = None, **kwargs) -> Dict:
        """Generate a response using OpenAI API.
        
        Args:
            query: User's query
            context: Retrieved knowledge contexts
            
        Returns:
            Response data
        """
        if not self.client:
            return {"response": "OpenAI API key is not configured.", "confidence": 0.0}
        
        try:
            # Extract context for prompt
            context_text = "\n\n".join([item.get("text", "") for item in (context or [])])
            
            system_message = f"""You are an AI assistant for Lucky Train project.
You provide helpful, accurate, and concise responses.
Use the following context to inform your answer:
{context_text}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            
            return {
                "response": result,
                "confidence": 0.9,
                "model_type": "OpenAI",
                "model": self.model
            }
        except Exception as e:
            logger.error(f"Error in OpenAI response generation: {e}")
            return {"response": "I encountered an error processing your request.", "confidence": 0.0}
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities

def create_ai_model(model_type: str, config: Dict = None) -> BaseAIModel:
    """Factory function to create an AI model instance.
    
    Args:
        model_type: Type of the AI model to create
        config: Configuration for the model
        
    Returns:
        AI model instance
    """
    model_type = model_type.lower()
    
    if model_type == "openai":
        return OpenAIModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 