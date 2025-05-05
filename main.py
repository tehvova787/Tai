import argparse
import atexit
from system import init_system
from utils import run_telegram_bot, run_web_interface, run_metaverse_assistant, run_console, run_all, run_blockchain_demo, run_multimodal_demo, run_dataset_demo, check_environment, ensure_directories

def main():
    """Main function to parse command-line arguments and run the appropriate component."""
    global system
    
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