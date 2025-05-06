# Lucky Train AI Deployment Setup

This document provides instructions for setting up and deploying the Lucky Train AI web interface.

## Environment Variables

The application requires certain environment variables to function properly. You can configure these in your hosting environment (like Render.com) or locally in a `.env` file.

### Required Environment Variables

```
# OpenAI API Key - Required for full AI functionality
OPENAI_API_KEY=your_openai_api_key_here

# Web server configuration
PORT=10000  # Used by hosting platforms like Render.com
```

### Optional Environment Variables

```
# Flask secret key for session management
FLASK_SECRET_KEY=a_random_secret_key_here

# Debug mode
DEBUG=True

# Logging level
LOG_LEVEL=INFO
```

## Setting Up on Render.com

1. Create a new Web Service in your Render dashboard
2. Connect your repository
3. Configure these settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python src/web_interface_demo.py`
4. Add your environment variables in the "Environment" section:
   - Add `OPENAI_API_KEY` with your actual API key
   - The `PORT` variable is automatically set by Render

## Local Development

For local development:

1. Create a `.env` file in the project root with the necessary environment variables
2. Run the application with `python src/web_interface_demo.py`

## Troubleshooting

### "No OpenAI API key provided" Warning

If you see the warning "No OpenAI API key provided. AI functionality will be limited to demo responses", it means:

1. The application is running but couldn't find a valid OpenAI API key
2. The AI functionality is running in demo mode with predefined responses

To fix this:
- Make sure you've added a valid `OPENAI_API_KEY` to your environment variables
- If using a `.env` file, ensure it's properly formatted and located in the project root
- Restart the application after adding the key 