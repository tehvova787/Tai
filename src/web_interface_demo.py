"""
Simplified Web Interface Demo for Lucky Train AI Assistant

This module provides a simplified web interface for demonstrating the new LuckyTrainAI UI.
"""

import os
import logging
from datetime import datetime
import time
import uuid

from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory, session
from flask_cors import CORS
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LuckyTrainWebInterfaceDemo:
    """Simplified web interface demo for the Lucky Train AI assistant."""
    
    def __init__(self, openai_api_key=None):
        """Initialize the web interface demo.
        
        Args:
            openai_api_key: Optional OpenAI API key. If not provided, will try to get from environment variable.
        """
        # Initialize Flask app
        self.app = Flask(__name__, static_folder="web/static", template_folder="web/templates")
        CORS(self.app)  # Enable CORS for all routes
        
        # Set up sessions
        self.sessions = {}
        
        # Set secret key for flask sessions from environment variable
        self.app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
        
        # Initialize OpenAI API key
        self.openai_api_key = None
        if openai_api_key and '\0' not in openai_api_key:
            self.openai_api_key = openai_api_key
        elif os.environ.get("OPENAI_API_KEY") and '\0' not in os.environ.get("OPENAI_API_KEY", ""):
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
            
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("OpenAI API key configured successfully")
        else:
            logger.warning("No OpenAI API key provided. AI functionality will be limited to demo responses.")
        
        # Set up routes
        self._setup_routes()
        
        logger.info("Web interface demo initialized successfully")
    
    def _setup_routes(self):
        """Set up the routes for the web interface."""
        # Main routes
        self.app.route("/")(self.index)
        self.app.route("/chat")(self.chat_page)
        self.app.route("/test")(self.test_page)
        self.app.route("/simple")(self.simple_page)
        
        # LuckyTrainAI UI routes
        self.app.route("/luckytrainai")(self.luckytrainai_index)
        self.app.route("/luckytrainai/chat")(self.luckytrainai_chat)
        
        # API routes
        self.app.route("/api/chat", methods=["POST"])(self.chat_api)
        self.app.route("/api/settings", methods=["GET", "POST"])(self.settings_api)
        
        # Static files and media
        self.app.route("/static/<path:path>")(self.serve_static)
        self.app.route("/media/<path:path>")(self.serve_media)
    
    def _create_session(self) -> str:
        """Create a new session.
        
        Returns:
            The session ID.
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": time.time(),
            "last_active": time.time(),
            "messages": []
        }
        return session_id
    
    def _get_session(self, session_id: str):
        """Get a session by ID.
        
        Args:
            session_id: The session ID.
            
        Returns:
            The session data, or None if not found.
        """
        session = self.sessions.get(session_id)
        if session:
            session["last_active"] = time.time()
        return session
    
    def index(self):
        """Handle the root route.
        
        Returns:
            The rendered index page.
        """
        return render_template(
            "simple.html",
            title="Lucky Train AI Assistant Demo"
        )
    
    def chat_page(self):
        """Handle the chat page route.
        
        Returns:
            The rendered chat page.
        """
        return render_template(
            "chat.html",
            title="Lucky Train AI Assistant Demo - Chat",
            theme="light",
            welcome_message="Привет! Я демо-версия AI-ассистента проекта Lucky Train. Чем я могу вам помочь?"
        )
    
    def test_page(self):
        """Handle the test page route.
        
        Returns:
            The rendered test page.
        """
        return render_template(
            "test.html",
            title="Test Page"
        )
    
    def simple_page(self):
        """Handle the simple page route.
        
        Returns:
            The rendered simple page.
        """
        return render_template(
            "simple.html",
            title="Simple Lucky Train AI Page"
        )
    
    def luckytrainai_index(self):
        """Handle the LuckyTrainAI index route.
        
        Returns:
            The rendered LuckyTrainAI index page.
        """
        return render_template(
            "luckytrainai.html",
            title="LuckyTrainAI Demo",
            theme="dark",
            welcome_message="Привет! Я демо-версия AI-ассистента проекта Lucky Train. Чем я могу вам помочь?"
        )
    
    def luckytrainai_chat(self):
        """Handle the LuckyTrainAI chat route.
        
        Returns:
            The rendered LuckyTrainAI chat page.
        """
        return render_template(
            "luckytrainai-chat.html",
            title="LuckyTrainAI Demo - Чат",
            theme="dark",
            welcome_message="Привет! Я демо-версия AI-ассистента проекта Lucky Train. Чем я могу вам помочь?"
        )
    
    def settings_api(self):
        """Handle the settings API route.
        
        Returns:
            JSON response with the current settings or updates settings.
        """
        if request.method == "GET":
            # Return current settings (without exposing API key)
            return jsonify({
                "has_openai_api_key": bool(self.openai_api_key)
            })
        elif request.method == "POST":
            # Update settings
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            # Update OpenAI API key if provided
            api_key = data.get("openai_api_key")
            if api_key and '\0' not in api_key:
                self.openai_api_key = api_key
                openai.api_key = api_key
                logger.info("OpenAI API key updated")
                
                # Don't store API key in environment to avoid exposing it
                # Just keep it in memory
                
                return jsonify({
                    "success": True,
                    "message": "API key updated successfully"
                })
            
            return jsonify({
                "error": "No settings were updated"
            }), 400
    
    def chat_api(self):
        """Handle the chat API route.
        
        Returns:
            JSON response with the assistant's reply.
        """
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        message = data.get("message")
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        session_id = data.get("session_id")
        if not session_id:
            session_id = self._create_session()
        elif session_id not in self.sessions:
            session_id = self._create_session()
        
        # Get user info
        user_id = data.get("user_id", f"web_user_{session_id}")
        
        # Handle the message
        try:
            start_time = time.time()
            
            # Generate response using OpenAI if available, otherwise use demo
            if self.openai_api_key:
                response = self._generate_openai_response(message, session_id)
            else:
                response = self._generate_demo_response(message)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update session with the message and response
            session = self._get_session(session_id)
            if session:
                session["messages"].append({
                    "role": "user",
                    "content": message,
                    "timestamp": time.time()
                })
                session["messages"].append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": time.time()
                })
            
            # Create message ID for feedback
            message_id = f"web_{session_id}_{int(time.time())}"
            
            return jsonify({
                "session_id": session_id,
                "message_id": message_id,
                "response": response,
                "response_time": response_time
            })
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    
    def _generate_openai_response(self, message, session_id):
        """Generate a response using OpenAI.
        
        Args:
            message: The user's message.
            session_id: The session ID.
            
        Returns:
            The generated response.
        """
        session = self._get_session(session_id)
        
        # Prepare conversation history
        messages = []
        if session and session["messages"]:
            # Add up to 10 most recent messages for context
            for msg in session["messages"][-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add system message for context
        messages.insert(0, {
            "role": "system",
            "content": "Ты AI-ассистент проекта Lucky Train на блокчейне TON. Твоя задача - отвечать на вопросы о проекте, токене LTT, метавселенной и блокчейне. Будь дружелюбным и информативным."
        })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        try:
            # Call OpenAI API
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            # Extract response
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Fallback to demo response if API fails
            return self._generate_demo_response(message)
    
    def _generate_demo_response(self, message):
        """Generate a demo response based on the message content.
        
        Args:
            message: The user's message.
            
        Returns:
            A demo response.
        """
        message_lower = message.lower()
        
        if "привет" in message_lower or "здравствуй" in message_lower:
            return "Привет! Рад видеть вас в демо-версии LuckyTrainAI. Как я могу помочь?"
        
        if "блокчейн" in message_lower or "ton" in message_lower:
            return "LuckyTrain использует блокчейн TON (The Open Network) для своей экосистемы. TON обеспечивает высокую скорость транзакций и низкие комиссии, что делает его идеальным для игровых и метавселенных проектов."
        
        if "метавселенн" in message_lower:
            return "Метавселенная LuckyTrain - это виртуальный мир с элементами игры, где пользователи могут взаимодействовать, строить и участвовать в экономической системе. Экономика метавселенной основана на токене LTT, который можно использовать внутри экосистемы для различных целей."
        
        if "токен" in message_lower or "ltt" in message_lower:
            return "Токен LTT (Lucky Train Token) - это основная валюта экосистемы LuckyTrain. Он имеет дефляционную модель и используется для транзакций внутри метавселенной, стейкинга и участия в управлении проектом."
        
        if "карт" in message_lower:
            return "Дорожная карта проекта Lucky Train включает в себя несколько фаз: разработку базовой экосистемы, запуск токена LTT, создание метавселенной и интеграцию с другими блокчейн-проектами. Каждая фаза содержит множество задач, направленных на развитие проекта."
        
        if "команд" in message_lower:
            return "Команда LuckyTrain состоит из опытных разработчиков, дизайнеров и блокчейн-экспертов. Все участники команды имеют большой опыт в создании цифровых продуктов и работе с блокчейн-технологиями."
        
        # Default response
        return "Как демо-версия, я могу рассказать о проекте LuckyTrain, его токене, метавселенной и блокчейне TON. Спросите меня об этих темах, чтобы узнать больше."
    
    def serve_static(self, path):
        """Serve static files.
        
        Args:
            path: The path to the static file.
            
        Returns:
            The static file.
        """
        return send_from_directory(self.app.static_folder, path)
    
    def serve_media(self, path):
        """Serve media files.
        
        Args:
            path: The path to the media file.
            
        Returns:
            The media file.
        """
        media_path = os.path.join(self.app.static_folder, "media")
        return send_from_directory(media_path, path)
    
    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Run the web interface.
        
        Args:
            host: The host to run the server on.
            port: The port to run the server on.
            debug: Whether to run in debug mode.
        """
        logger.info(f"Starting web interface demo on {host}:{port}")
        # Disable Flask's automatic environment loading
        self.app.config['ENV'] = 'development'
        self.app.run(host=host, port=port, debug=debug, load_dotenv=False)

# Run the web interface if executed directly
if __name__ == "__main__":
    # Initialize and run the web interface without trying to get API key from environment
    # This avoids potential issues with null characters in environment variables
    web_interface = LuckyTrainWebInterfaceDemo(openai_api_key=None)
    web_interface.run(host="127.0.0.1", port=5000, debug=True) 