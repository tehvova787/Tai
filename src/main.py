"""
Main script for Lucky Train AI Assistant

This script provides a command-line interface to run different components
of the Lucky Train AI assistant system.
"""

import argparse
import logging
import os
import sys
import threading
import atexit
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Import the system initialization module
from system_init import init_system, get_system

# Load environment variables
load_dotenv()

# Add the bot directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bot'))

# Initialize the system with default config
system = init_system()
logger = system.logger

def ensure_directories():
    """Ensure that required directories exist."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

def run_telegram_bot(config_path: str, ai_model: str = None):
    """Run the Telegram bot.
    
    Args:
        config_path: Path to the configuration file.
        ai_model: AI model type to use.
    """
    from bot.telegram_bot import LuckyTrainTelegramBot
    
    # Get the Telegram bot token from environment variable
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set")
        print("Please set the TELEGRAM_BOT_TOKEN environment variable and try again.")
        return
    
    logger.info("Starting Telegram bot...")
    bot = LuckyTrainTelegramBot(token, config_path, ai_model)
    bot.run()

def run_web_interface(config_path: str, host: str = '0.0.0.0', port: int = 5000, ai_model: str = None):
    """Run the web interface.
    
    Args:
        config_path: Path to the configuration file.
        host: The host to run on.
        port: The port to run on.
        ai_model: AI model type to use.
    """
    from web_interface import LuckyTrainWebInterface
    
    logger.info(f"Starting web interface on {host}:{port}...")
    interface = LuckyTrainWebInterface(config_path, ai_model)
    interface.run(host=host, port=port, debug=False)

def run_metaverse_assistant(config_path: str):
    """Run the metaverse assistant in interactive mode.
    
    Args:
        config_path: Path to the configuration file.
    """
    from metaverse_integration import LuckyTrainMetaverseAssistant
    
    logger.info("Starting metaverse assistant in interactive mode...")
    assistant = LuckyTrainMetaverseAssistant(config_path)
    assistant.run_interactive_console()

def run_console(config_path: str, ai_model: str = None):
    """Run the assistant in console mode for testing.
    
    Args:
        config_path: Path to the configuration file.
        ai_model: AI model type to use.
    """
    from bot.assistant import LuckyTrainAssistant
    
    logger.info("Starting assistant in console mode...")
    assistant = LuckyTrainAssistant(config_path)
    
    # Set AI model if specified
    if ai_model and assistant.use_ai_system:
        if assistant.ai_system.set_ai_model(ai_model):
            print(f"Using AI model: {ai_model.upper()}")
        else:
            print(f"Failed to set AI model: {ai_model}. Using default model.")
    
    print("Lucky Train AI Assistant - Console Mode")
    print("Commands:")
    print("  /exit - Exit the console")
    print("  /models - List available AI models")
    print("  /model <name> - Switch to specified AI model")
    print("  /dbs - List available database connectors")
    print("  /db <name> <query> - Execute a query on the specified database")
    print()
    
    while True:
        try:
            user_input = input("> ")
            
            if user_input.lower() == '/exit':
                break
            
            elif user_input.lower() == '/models' and assistant.use_ai_system:
                models = assistant.ai_system.get_available_models()
                current = assistant.ai_system.current_model_type
                print(f"Available models: {', '.join(models)}")
                print(f"Current model: {current}")
                continue
            
            elif user_input.lower().startswith('/model ') and assistant.use_ai_system:
                model_name = user_input.replace('/model ', '').strip().lower()
                if assistant.ai_system.set_ai_model(model_name):
                    print(f"Switched to model: {model_name}")
                else:
                    print(f"Failed to switch to model: {model_name}")
                continue
            
            elif user_input.lower() == '/dbs' and assistant.use_ai_system:
                dbs = assistant.ai_system.get_available_connectors()
                print(f"Available database connectors: {', '.join(dbs) if dbs else 'None'}")
                continue
            
            response = assistant.handle_message(user_input, "console_user", "console")
            print(response)
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in console mode: {e}")
            print(f"Error: {e}")

def run_all(config_path: str, web_port: int = 5000, ai_model: str = None):
    """Run all components in separate threads.
    
    Args:
        config_path: Path to the configuration file.
        web_port: The port for the web interface.
        ai_model: AI model type to use.
    """
    # Create and start threads for each component
    telegram_thread = threading.Thread(
        target=run_telegram_bot,
        args=(config_path, ai_model),
        name="TelegramBotThread"
    )
    
    web_thread = threading.Thread(
        target=run_web_interface,
        args=(config_path, '0.0.0.0', web_port, ai_model),
        name="WebInterfaceThread"
    )
    
    # Set threads as daemon so they exit when the main thread exits
    telegram_thread.daemon = True
    web_thread.daemon = True
    
    # Start the threads
    telegram_thread.start()
    web_thread.start()
    
    logger.info("All components started. Press Ctrl+C to exit.")
    
    try:
        # Keep the main thread running
        while True:
            telegram_thread.join(1)
            web_thread.join(1)
            
            # If both threads have exited, exit the program
            if not telegram_thread.is_alive() and not web_thread.is_alive():
                logger.info("All component threads have exited. Shutting down.")
                break
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Shutting down.")

def check_environment():
    """Check if required environment variables are set."""
    missing_vars = []
    
    # Check for Telegram bot token if running telegram component
    if 'telegram' in sys.argv or 'all' in sys.argv:
        if not os.getenv("TELEGRAM_BOT_TOKEN"):
            missing_vars.append("TELEGRAM_BOT_TOKEN")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set. The assistant will fall back to TF-IDF.")
    
    # Check for TON API key
    if not os.getenv("TON_API_KEY"):
        logger.warning("TON_API_KEY environment variable not set. Blockchain features will be limited.")

    # Check for ElevenLabs API key
    if not os.getenv("ELEVENLABS_API_KEY"):
        logger.warning("ELEVENLABS_API_KEY environment variable not set. Voice features will use OpenAI TTS or be limited.")
    
    # If there are missing required variables, print a message and exit
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment and try again.")
        sys.exit(1)

def main():
    """Main function to parse command-line arguments and run the appropriate component."""
    # Register shutdown handler
    atexit.register(system.shutdown)
    
    parser = argparse.ArgumentParser(description="Lucky Train AI Assistant")
    
    # Command-line arguments
    parser.add_argument(
        'component',
        choices=['telegram', 'web', 'metaverse', 'console', 'all', 'blockchain', 'multimodal', 'dataset'],
        help="The component to run"
    )
    
    parser.add_argument(
        '--config',
        default='./config/config.json',
        help="Path to the configuration file (default: ./config/config.json)"
    )
    
    parser.add_argument(
        '--web-port',
        type=int,
        default=5000,
        help="Port for the web interface (default: 5000)"
    )
    
    parser.add_argument(
        '--web-host',
        default='0.0.0.0',
        help="Host for the web interface (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        '--language',
        default=None,
        help="Force a specific language (e.g., en, ru, es, zh)"
    )
    
    # Add AI model type argument
    parser.add_argument(
        '--ai-model',
        choices=[
            'ani', 'agi', 'asi', 
            'machine_learning', 'deep_learning', 'reinforcement_learning',
            'analytical_ai', 'interactive_ai', 'functional_ai',
            'symbolic_systems', 'connectionist_systems', 'hybrid_systems'
        ],
        default=None,
        help="Specify the AI model type to use"
    )
    
    # Add database connector argument
    parser.add_argument(
        '--db-connector',
        choices=['chat2db', 'bigquery', 'aurora', 'cosmosdb', 'snowflake', 'db2ai'],
        default=None,
        help="Specify the database connector to use"
    )
    
    # Add dataset connector argument
    parser.add_argument(
        '--dataset',
        choices=['kaggle', 'google_dataset_search', 'uci_ml', 'imagenet', 'common_crawl', 
                 'huggingface', 'datagov', 'zenodo', 'arxiv'],
        default=None,
        help="Specify the dataset connector to use for the dataset demo"
    )
    
    # Add security flags
    parser.add_argument(
        '--secure-mode',
        action='store_true',
        help="Enable enhanced security features"
    )
    
    # Add database flags
    parser.add_argument(
        '--db-path',
        default=None,
        help="Path to the database file"
    )
    
    # Add cache flags
    parser.add_argument(
        '--cache-mode',
        choices=['memory', 'disk', 'none'],
        default=None,
        help="Caching mode to use"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Re-initialize system with specified config if provided
    if args.config != './config/config.json':
        global system
        system = init_system(args.config)
        logger = system.logger
    
    # Update config based on command line arguments
    if args.secure_mode:
        system.config['security_settings']['enhanced_security'] = True
    
    if args.db_path:
        system.config['database_settings']['db_path'] = args.db_path
    
    if args.cache_mode:
        if args.cache_mode == 'none':
            system.config['cache_settings']['enabled'] = False
        else:
            system.config['cache_settings']['type'] = args.cache_mode
            system.config['cache_settings']['enabled'] = True
    
    # Check environment variables
    check_environment()
    
    # Ensure required directories exist
    ensure_directories()
    
    # Run the selected component
    if args.component == 'telegram':
        run_telegram_bot(args.config, args.ai_model)
    elif args.component == 'web':
        run_web_interface(args.config, args.web_host, args.web_port, args.ai_model)
    elif args.component == 'metaverse':
        run_metaverse_assistant(args.config)
    elif args.component == 'console':
        run_console(args.config, args.ai_model)
    elif args.component == 'all':
        run_all(args.config, args.web_port, args.ai_model)
    elif args.component == 'blockchain':
        run_blockchain_demo(args.config)
    elif args.component == 'multimodal':
        run_multimodal_demo(args.config)
    elif args.component == 'dataset':
        run_dataset_demo(args.config, args.dataset)

def run_blockchain_demo(config_path: str):
    """Run a demonstration of the blockchain integration features.
    
    Args:
        config_path: Path to the configuration file.
    """
    from blockchain_integration import TONBlockchainIntegration
    
    logger.info("Starting blockchain integration demo...")
    blockchain = TONBlockchainIntegration(config_path)
    
    print("Lucky Train AI - Blockchain Integration Demo")
    print("-------------------------------------------")
    print()
    
    # Get blockchain info
    print("Getting blockchain info...")
    info = blockchain.get_blockchain_info()
    print(f"Blockchain Info: {info}")
    print()
    
    # Get TON price
    print("Getting TON price...")
    ton_price = blockchain.get_market_price("TON")
    print(f"TON Price: {ton_price}")
    print()
    
    # Get LTT price (placeholder)
    print("Getting LTT price...")
    ltt_price = blockchain.get_market_price("LTT")
    print(f"LTT Price: {ltt_price}")
    print()
    
    # Demo wallet authentication
    print("Wallet Authentication Demo:")
    user_id = "demo_user"
    message, timestamp = blockchain.get_wallet_auth_message(user_id)
    print(f"Authentication Message: {message}")
    print(f"This message would be signed by the user's TON wallet")
    print()
    
    # Generate a deep link example
    print("TON Wallet Deep Link Example:")
    deep_link = blockchain.generate_deep_link("transfer", {
        "address": "EQD__________________________________________0",
        "amount": 10,
        "comment": "Purchase Lucky Train NFT #123"
    })
    print(f"Deep Link: {deep_link}")
    print()
    
    print("Demo Complete. Press Enter to exit.")
    input()

def run_multimodal_demo(config_path: str):
    """Run a demonstration of the multimodal interface features.
    
    Args:
        config_path: Path to the configuration file.
    """
    from multimodal_interface import MultimodalInterface
    import base64
    
    logger.info("Starting multimodal interface demo...")
    interface = MultimodalInterface(config_path)
    
    print("Lucky Train AI - Multimodal Interface Demo")
    print("------------------------------------------")
    print()
    
    # Text-to-speech demo
    print("Text-to-Speech Demo:")
    texts = {
        "ru": "Добро пожаловать в метавселенную Lucky Train! Здесь вы можете исследовать множество уникальных локаций.",
        "en": "Welcome to the Lucky Train metaverse! Here you can explore many unique locations.",
        "es": "¡Bienvenido al metaverso de Lucky Train! Aquí puedes explorar muchas ubicaciones únicas."
    }
    
    for lang, text in texts.items():
        print(f"\nGenerating speech in {lang}...")
        result = interface.text_to_speech(text, lang, emotion="happy")
        
        if result["success"]:
            try:
                audio_file = f"demo_speech_{lang}.mp3"
                with open(audio_file, "wb") as f:
                    f.write(base64.b64decode(result["audio_base64"]))
                print(f"Speech saved to {audio_file}")
            except Exception as e:
                print(f"Error saving audio file: {e}")
        else:
            print(f"Error generating speech: {result.get('error')}")
    
    print("\nImage Generation Demo:")
    print("Generating an image of Central Station in the Lucky Train metaverse...")
    
    result = interface.generate_location_preview("Central Station", "station")
    
    if result["success"]:
        try:
            image_file = "demo_central_station.png"
            with open(image_file, "wb") as f:
                f.write(base64.b64decode(result["image_base64"]))
            print(f"Image saved to {image_file}")
        except Exception as e:
            print(f"Error saving image file: {e}")
    else:
        print(f"Error generating image: {result.get('error')}")
    
    print("\nNFT AR Preview Demo:")
    print("Creating an AR preview of a Lucky Train NFT...")
    
    nft_data = {
        "id": "123",
        "name": "Lucky Express #123",
        "description": "A unique digital train from the Lucky Train metaverse.",
        "price": "100",
        "owner": "EQD__________________________________________0"
    }
    
    result = interface.create_ar_preview(nft_data)
    
    if result["success"]:
        try:
            image_file = "demo_nft_preview.png"
            with open(image_file, "wb") as f:
                f.write(base64.b64decode(result["image_base64"]))
            print(f"NFT preview saved to {image_file}")
        except Exception as e:
            print(f"Error saving NFT preview file: {e}")
    else:
        print(f"Error creating NFT preview: {result.get('error')}")
    
    print("\nDemo Complete. Press Enter to exit.")
    input()

def run_dataset_demo(config_path: str, dataset_type: str = None):
    """Run a demonstration of the dataset connectors.
    
    Args:
        config_path: Path to the configuration file.
        dataset_type: The specific dataset connector to use.
    """
    try:
        import json
        from dataset_connectors import create_dataset_connector
        
        logger.info("Starting dataset connectors demo...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("Lucky Train AI - Dataset Connectors Demo")
        print("----------------------------------------")
        print()
        
        # Initialize dataset connectors
        dataset_configs = config.get("data_source_settings", {})
        
        # Use specified dataset connector
        if dataset_type:
            settings = dataset_configs.get(dataset_type, {})
            print(f"Testing {dataset_type} connector...")
            
            connector = create_dataset_connector(dataset_type, settings)
            if connector and connector.connect():
                print(f"Successfully connected to {dataset_type}")
                
                # Search
                search_query = input("Enter search query (or press Enter for default 'machine learning'): ") or "machine learning"
                limit = int(input("Enter maximum number of results (or press Enter for default 5): ") or "5")
                
                print(f"\nSearching for '{search_query}'...")
                search_results = connector.search(search_query, limit=limit)
                
                if search_results.get("success"):
                    data = search_results.get("data", [])
                    print(f"\nFound {len(data)} results:")
                    
                    for i, item in enumerate(data):
                        if isinstance(item, dict):
                            title = item.get("title") or item.get("name") or item.get("id")
                            print(f"{i+1}. {title}")
                    
                    # Ask if user wants to download a dataset
                    if data:
                        download = input("\nDo you want to download a dataset? (y/n): ").lower() == 'y'
                        
                        if download:
                            idx = int(input(f"Enter the number of the dataset to download (1-{len(data)}): ")) - 1
                            
                            if 0 <= idx < len(data):
                                dataset_id = data[idx].get("id")
                                print(f"\nDownloading dataset ID: {dataset_id}")
                                
                                download_result = connector.download(dataset_id)
                                
                                if download_result.get("success"):
                                    print(f"Successfully downloaded dataset to: {download_result.get('path')}")
                                    if "files" in download_result:
                                        print(f"Files: {download_result.get('files')}")
                                else:
                                    print(f"Error downloading dataset: {download_result.get('error')}")
                else:
                    print(f"Error searching: {search_results.get('error')}")
            else:
                print(f"Failed to connect to {dataset_type}")
        
        # Test all enabled dataset connectors
        else:
            connectors = initialize_dataset_connectors(config)
            
            if not connectors:
                print("No dataset connectors were initialized. Check your configuration.")
            else:
                print(f"Initialized {len(connectors)} dataset connectors:")
                
                for i, (name, connector) in enumerate(connectors.items()):
                    print(f"{i+1}. {name}")
                
                # Choose a connector to test
                idx = int(input("\nEnter the number of the connector to test: ")) - 1
                
                if 0 <= idx < len(connectors):
                    name, connector = list(connectors.items())[idx]
                    
                    print(f"\nTesting {name} connector...")
                    
                    # Demo search
                    search_query = input("Enter search query (or press Enter for default 'machine learning'): ") or "machine learning"
                    print(f"\nSearching for '{search_query}'...")
                    
                    search_results = connector.search(search_query, limit=5)
                    
                    if search_results.get("success"):
                        data = search_results.get("data", [])
                        print(f"\nFound {len(data)} results")
                        
                        for i, item in enumerate(data):
                            if isinstance(item, dict):
                                title = item.get("title") or item.get("name") or item.get("id")
                                print(f"{i+1}. {title}")
                    else:
                        print(f"Error searching: {search_results.get('error')}")
                else:
                    print("Invalid selection.")
        
        print("\nDemo Complete. Press Enter to exit.")
        input()
        
    except ImportError as e:
        logger.error(f"Error importing dataset_connectors module: {e}")
        print(f"Error: Could not import dataset_connectors module. Make sure it's installed.")
        print(f"Import error: {e}")
    except Exception as e:
        logger.error(f"Error in dataset connectors demo: {e}")
        print(f"Error: {e}")

def initialize_dataset_connectors(config):
    """Initialize dataset connectors from configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        Dictionary of initialized dataset connectors
    """
    try:
        from dataset_connectors import create_dataset_connector
        
        dataset_configs = config.get("data_source_settings", {})
        connectors = {}
        
        for dataset_type, settings in dataset_configs.items():
            if settings.get("enabled", False):
                logger.info(f"Initializing {dataset_type} dataset connector")
                connector = create_dataset_connector(dataset_type, settings)
                if connector:
                    connectors[dataset_type] = connector
        
        return connectors
    except ImportError:
        logger.warning("dataset_connectors module not available")
        return {}
    except Exception as e:
        logger.error(f"Error initializing dataset connectors: {str(e)}")
        return {}

if __name__ == "__main__":
    main() 