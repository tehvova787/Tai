#!/usr/bin/env python3
"""
OpenAI Integration Test for Lucky Train AI Assistant (Python Version)

This script tests the OpenAI API connection with the provided API keys
"""

import os
import sys
import logging
import re
from typing import Dict, Any
import openai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_openai_connection():
    """Test the connection to OpenAI API using environment variables."""
    try:
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
            
        # Initialize OpenAI client
        client = openai.OpenAI(
            api_key=api_key,
            organization=os.getenv("OPENAI_ORGANIZATION_ID")
        )
        
        # Test the connection
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            temperature=0.7
        )
        
        logger.info("OpenAI API connection successful!")
        logger.info(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to OpenAI API: {str(e)}")
        return False

def update_env_file():
    """Update the .env file with the current API key."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        org_id = os.getenv("OPENAI_ORGANIZATION_ID")
        
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            return False
            
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
            if org_id:
                f.write(f"OPENAI_ORGANIZATION_ID={org_id}\n")
                
        logger.info("Successfully updated .env file")
        return True
        
    except Exception as e:
        logger.error(f"Error updating .env file: {str(e)}")
        return False

def update_ai_models_py():
    """Update the ai_models.py file to use environment variables."""
    try:
        with open("src/ai_models.py", "r") as f:
            content = f.read()
            
        # Replace hardcoded API key with environment variable
        updated_content = re.sub(
            r'self\.openai_api_key = "sk\-[a-zA-Z0-9_\-]+"',
            'self.openai_api_key = os.getenv("OPENAI_API_KEY")',
            content
        )
        
        with open("src/ai_models.py", "w") as f:
            f.write(updated_content)
            
        logger.info("Successfully updated ai_models.py")
        return True
        
    except Exception as e:
        logger.error(f"Error updating ai_models.py: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the OpenAI connection
    test_result = test_openai_connection()
    
    if test_result:
        print("\n✅ OpenAI API connection test successful!")
        
        # Update configuration files
        update_env_file()
        update_ai_models_py()
        
        print("✅ Configuration files updated successfully!")
        
        sys.exit(0)
    else:
        print("\n❌ OpenAI API connection test failed!")
        sys.exit(1) 