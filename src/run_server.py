"""
Script to run the LuckyTrainAI web interface without using dotenv loading.
"""

import os
import sys
from web_interface_demo import LuckyTrainWebInterfaceDemo

if __name__ == "__main__":
    # Disable dotenv auto-loading in Flask
    os.environ['FLASK_SKIP_DOTENV'] = '1'
    
    # Initialize and run the web interface
    web_interface = LuckyTrainWebInterfaceDemo(openai_api_key=None)
    
    # Get port from command line arguments or use default
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port 5000.")
    
    try:
        web_interface.run(host="0.0.0.0", port=port, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        
        # Try alternative port if the first one failed
        if port == 5000:
            try:
                print("Trying alternative port 8080...")
                web_interface.run(host="0.0.0.0", port=8080, debug=True)
            except Exception as e2:
                print(f"Error starting server on alternative port: {e2}") 