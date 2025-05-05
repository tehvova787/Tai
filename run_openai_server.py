"""
Lucky Train AI with OpenAI Integration

This script runs the Lucky Train AI web server with OpenAI API integration.
"""

import os
import sys
from server import app

def main():
    """Run the web server with OpenAI integration."""
    
    # Get API key from environment, with a fallback to the default value
    api_key = os.environ.get('OPENAI_API_KEY')
    org_id = os.environ.get('OPENAI_ORGANIZATION_ID')
    
    # Print information about the OpenAI configuration
    print("=" * 50)
    print("Lucky Train AI with OpenAI Integration")
    print("=" * 50)
    
    if not api_key or api_key.startswith('sk-proj-'):
        print("\033[93mWarning: Using default or test API key.\033[0m")
        print("For production use, set your own API key with:")
        print("  Windows: set OPENAI_API_KEY=your_api_key")
        print("  Linux/Mac: export OPENAI_API_KEY=your_api_key")
    else:
        print("OpenAI API key is configured.")
    
    if not org_id:
        print("\033[93mWarning: No organization ID set.\033[0m")
        print("If you have an organization ID, set it with:")
        print("  Windows: set OPENAI_ORGANIZATION_ID=your_org_id")
        print("  Linux/Mac: export OPENAI_ORGANIZATION_ID=your_org_id")
    else:
        print("OpenAI organization ID is configured.")
    
    # Check if 'requests' module is installed
    try:
        import requests
    except ImportError:
        print("\033[91mError: The 'requests' module is not installed.\033[0m")
        print("Install it with: pip install requests")
        return 1
    
    # Default port
    port = 5000
    
    # Print server information
    print("\nStarting server:")
    print(f"  URL: http://localhost:{port}")
    print("  Endpoints:")
    print("    - /: Main page")
    print("    - /homepage: Homepage")
    print("    - /api/openai/chat: OpenAI API endpoint for chat")
    
    print("\nPress Ctrl+C to stop the server.")
    print("=" * 50)
    
    # Run the server
    app.run(host='0.0.0.0', port=port, debug=True, load_dotenv=False)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 