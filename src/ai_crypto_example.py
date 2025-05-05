"""
Example Script for AI Models and Crypto Tools Integration

This script demonstrates how to use the AI model and crypto tool integrations
for Lucky Train.
"""

import os
import json
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta

# Import our integrations
from ai_model_integrations import create_model_interface
from crypto_tools_integration import CryptoToolFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_ai_models():
    """Test various AI model integrations."""
    
    logger.info("Testing AI Model Integrations")
    logger.info("-" * 50)
    
    # Test OpenAI GPT models
    gpt_config = {"model": "gpt-3.5-turbo"}
    gpt_model = create_model_interface("openai", gpt_config)
    
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Testing OpenAI GPT model...")
        response = gpt_model.generate("What is Lucky Train?")
        logger.info(f"GPT Response: {response.get('text', 'No response')}")
    else:
        logger.warning("Skipping OpenAI test - API key not set")
    
    logger.info("-" * 50)
    
    # Test Claude models
    claude_config = {"model": "claude-3-sonnet-20240229"}
    claude_model = create_model_interface("anthropic", claude_config)
    
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Testing Anthropic Claude model...")
        response = claude_model.generate("Explain cryptocurrencies in simple terms.")
        logger.info(f"Claude Response: {response.get('text', 'No response')}")
    else:
        logger.warning("Skipping Claude test - API key not set")
    
    logger.info("-" * 50)
    
    # Test Hugging Face models (BERT)
    try:
        bert_config = {"model_name": "bert-base-uncased", "task": "embeddings"}
        bert_model = create_model_interface("huggingface", bert_config)
        
        logger.info("Testing BERT model for embeddings...")
        response = bert_model.generate("Generate embeddings for this text.")
        
        if "embeddings" in response:
            embedding_size = response.get("dimensions", 0)
            logger.info(f"BERT Embeddings generated with dimension: {embedding_size}")
        else:
            logger.info(f"BERT Response: {response}")
    except Exception as e:
        logger.warning(f"Skipping BERT test - Error: {e}")
    
    logger.info("-" * 50)
    
    # Test LLaMA model if available
    llama_model_path = os.path.join("models", "llama-2-7b-chat.gguf")
    if os.path.exists(llama_model_path):
        try:
            llama_config = {"model_path": llama_model_path}
            llama_model = create_model_interface("llama", llama_config)
            
            logger.info("Testing LLaMA model...")
            response = llama_model.generate("Tell me about blockchain technology.")
            logger.info(f"LLaMA Response: {response.get('text', 'No response')}")
        except Exception as e:
            logger.warning(f"Skipping LLaMA test - Error: {e}")
    else:
        logger.warning(f"Skipping LLaMA test - Model file not found at {llama_model_path}")
    
    logger.info("-" * 50)

def test_crypto_tools():
    """Test various crypto tool integrations."""
    
    logger.info("Testing Crypto Tool Integrations")
    logger.info("-" * 50)
    
    # Test Trality
    trality_config = {}
    trality_api = CryptoToolFactory.create_tool("trality", trality_config)
    
    logger.info("Testing Trality connection...")
    status = trality_api.get_status()
    logger.info(f"Trality Status: {status}")
    
    if status.get("status") == "connected":
        bots = trality_api.get_bots()
        logger.info(f"Trality Bots: {json.dumps(bots, indent=2)}")
    
    logger.info("-" * 50)
    
    # Test 3Commas
    three_commas_config = {}
    three_commas_api = CryptoToolFactory.create_tool("3commas", three_commas_config)
    
    logger.info("Testing 3Commas connection...")
    status = three_commas_api.get_status()
    logger.info(f"3Commas Status: {status}")
    
    if status.get("status") == "connected":
        bots = three_commas_api.get_bots()
        logger.info(f"3Commas Bots: {json.dumps(bots, indent=2)}")
    
    logger.info("-" * 50)
    
    # Test Glassnode
    glassnode_config = {}
    glassnode_api = CryptoToolFactory.create_tool("glassnode", glassnode_config)
    
    logger.info("Testing Glassnode connection...")
    status = glassnode_api.get_status()
    logger.info(f"Glassnode Status: {status}")
    
    if status.get("status") == "connected":
        # Get Bitcoin price data
        from_date = int((datetime.now() - timedelta(days=7)).timestamp())
        to_date = int(datetime.now().timestamp())
        
        btc_price = glassnode_api.get_metric("market/price_usd_close", "BTC", from_date, to_date)
        logger.info(f"Bitcoin Price Data: {json.dumps(btc_price.get('data', [])[:5], indent=2)}")
    
    logger.info("-" * 50)

def main():
    """Main function to run the examples."""
    
    logger.info("Starting AI and Crypto Integration Example")
    logger.info("=" * 50)
    
    # Check for environment variables
    required_vars = [
        "OPENAI_API_KEY", 
        "ANTHROPIC_API_KEY", 
        "TRALITY_API_KEY",
        "THREE_COMMAS_API_KEY",
        "THREE_COMMAS_API_SECRET",
        "CRYPTOHOPPER_API_KEY",
        "GLASSNODE_API_KEY",
        "SANTIMENT_API_KEY"
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            logger.warning(f"Environment variable {var} is not set")
    
    # Run examples
    test_ai_models()
    test_crypto_tools()
    
    logger.info("=" * 50)
    logger.info("Example completed")

if __name__ == "__main__":
    main() 