"""
Simplified Web Interface Demo for Lucky Train AI Assistant

This module provides a simplified web interface for demonstrating the new LuckyTrainAI UI.
"""

import os
import logging
from datetime import datetime
import time
import uuid
import json

# Import required Flask modules
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory, session
from flask_cors import CORS

# Make requests module optional - we don't actually use it directly in the demo mode
try:
    import requests
except ImportError:
    requests = None
    logging.warning("The 'requests' module is not available. Some features may be limited.")

# Make OpenAI module optional - we'll fall back to demo mode if it's not available
try:
    import openai
except ImportError:
    openai = None
    logging.warning("The 'openai' module is not available. AI functionality will be limited to demo responses.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LuckyTrainWebInterfaceDemo:
    """Simplified web interface demo for the Lucky Train AI assistant."""
    
    def __init__(self, openai_api_key=None, knowledge_base_path=None):
        """Initialize the web interface demo.
        
        Args:
            openai_api_key: Optional OpenAI API key. If not provided, will try to get from environment variable.
            knowledge_base_path: Optional path to knowledge base files. If not provided, will use default location.
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
        
        # Check if OpenAI module is available
        if openai is not None:
            if openai_api_key and openai_api_key.strip() != "" and '\0' not in openai_api_key and openai_api_key != "your_openai_api_key_here":
                self.openai_api_key = openai_api_key
            elif os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API_KEY").strip() != "" and '\0' not in os.environ.get("OPENAI_API_KEY", "") and os.environ.get("OPENAI_API_KEY") != "your_openai_api_key_here":
                self.openai_api_key = os.environ.get("OPENAI_API_KEY")
                
            if self.openai_api_key:
                openai.api_key = self.openai_api_key
                logger.info("OpenAI API key configured successfully")
            else:
                logger.warning("No valid OpenAI API key provided. AI functionality will be limited to demo responses.")
        else:
            logger.warning("OpenAI module not available. AI functionality will be limited to demo responses.")
        
        # Initialize knowledge base
        self.knowledge_base_path = knowledge_base_path or os.path.join(os.path.dirname(__file__), "knowledge_base")
        self.knowledge_base = self._load_knowledge_base()
        
        # Set up routes
        self._setup_routes()
        
        logger.info("Web interface demo initialized successfully")
    
    def _load_knowledge_base(self):
        """Load knowledge base files.
        
        Returns:
            Dictionary containing knowledge base information.
        """
        knowledge_base = {}
        
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists(self.knowledge_base_path):
            os.makedirs(self.knowledge_base_path)
            logger.info(f"Created knowledge base directory at {self.knowledge_base_path}")
            
            # Create a default knowledge base file with basic information
            default_kb_path = os.path.join(self.knowledge_base_path, "luckytrainai_info.json")
            default_knowledge = {
                "project_info": {
                    "name": "Lucky Train",
                    "blockchain": "TON (The Open Network)",
                    "token": "LTT (Lucky Train Token)",
                    "description": "A metaverse project built on TON blockchain with its own token economy."
                },
                "token_details": {
                    "name": "Lucky Train Token",
                    "symbol": "LTT",
                    "model": "Deflationary",
                    "uses": ["Metaverse transactions", "Staking", "Governance"]
                },
                "metaverse_info": {
                    "description": "Virtual world with gaming elements where users can interact, build, and participate in the economy based on LTT token."
                }
            }
            
            with open(default_kb_path, 'w', encoding='utf-8') as f:
                json.dump(default_knowledge, f, ensure_ascii=False, indent=2)
                
            logger.info("Created default knowledge base file")
        
        # Load all knowledge base files
        try:
            for filename in os.listdir(self.knowledge_base_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.knowledge_base_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        kb_data = json.load(f)
                        knowledge_base[filename.replace('.json', '')] = kb_data
                        logger.info(f"Loaded knowledge base file: {filename}")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
        
        return knowledge_base
    
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
        
        # Knowledge base management routes
        self.app.route("/api/knowledge", methods=["GET", "POST"])(self.knowledge_api)
        
        # Static files and media
        self.app.route("/static/<path:path>")(self.serve_static)
        self.app.route("/media/<path:path>")(self.serve_media)
        
        # Favicon route
        self.app.route("/favicon.ico")(self.serve_favicon)
    
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
            welcome_message="Привет! Я AI-ассистент проекта Lucky Train. Задайте мне вопрос о проекте, и я постараюсь дать вам детальный ответ."
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
            welcome_message="Привет! Я AI-ассистент проекта Lucky Train. Задайте мне вопрос о проекте, и я постараюсь дать вам детальный ответ на основе актуальной информации."
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
            welcome_message="Привет! Я AI-ассистент проекта Lucky Train. Задайте мне вопрос о проекте, и я постараюсь дать вам детальный ответ на основе актуальной информации."
        )
    
    def settings_api(self):
        """Handle the settings API route.
        
        Returns:
            JSON response with the current settings or updates settings.
        """
        if request.method == "GET":
            # Return current settings (without exposing API key)
            return jsonify({
                "has_openai_api_key": bool(self.openai_api_key),
                "has_knowledge_base": bool(self.knowledge_base)
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
                
                return jsonify({
                    "success": True,
                    "message": "API key updated successfully"
                })
            
            return jsonify({
                "error": "No settings were updated"
            }), 400
    
    def knowledge_api(self):
        """Handle knowledge base API requests.
        
        Returns:
            JSON response with knowledge base info or confirmation of updates.
        """
        if request.method == "GET":
            # Return list of knowledge base files (not the full content for security)
            kb_list = []
            for name, data in self.knowledge_base.items():
                kb_list.append({
                    "name": name,
                    "categories": list(data.keys()),
                    "size": len(json.dumps(data, ensure_ascii=False))
                })
            
            return jsonify({
                "knowledge_base": kb_list
            })
            
        elif request.method == "POST":
            # Add or update knowledge base entry
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
                
            name = data.get("name")
            content = data.get("content")
            
            if not name or not content:
                return jsonify({"error": "Name and content are required"}), 400
            
            # Sanitize filename
            name = "".join(c for c in name if c.isalnum() or c in "_-")
            
            # Save knowledge base file
            file_path = os.path.join(self.knowledge_base_path, f"{name}.json")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
                
                # Update in-memory knowledge base
                self.knowledge_base[name] = content
                
                logger.info(f"Knowledge base updated: {name}")
                
                return jsonify({
                    "success": True,
                    "message": "Knowledge base updated successfully"
                })
            except Exception as e:
                logger.error(f"Error updating knowledge base: {e}")
                return jsonify({
                    "error": f"Failed to update knowledge base: {str(e)}"
                }), 500
    
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
            if openai is not None and self.openai_api_key:
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
    
    def _generate_knowledge_context(self, message):
        """Generate context from knowledge base relevant to the user's message.
        
        Args:
            message: The user's message.
            
        Returns:
            String containing relevant knowledge base information.
        """
        if not self.knowledge_base:
            return ""
        
        context_parts = []
        
        # Simple keyword-based matching for relevant knowledge
        message_lower = message.lower()
        
        # Common keywords to look for in the knowledge base
        keywords = {
            "token": ["token", "ltt", "криптовалют", "монет", "токен"],
            "blockchain": ["блокчейн", "ton", "тон", "транзакц", "блок"],
            "metaverse": ["метавселенн", "виртуальн", "мир"],
            "roadmap": ["карт", "план", "дорожн", "roadmap"],
            "team": ["команд", "сотрудник", "разработчик"]
        }
        
        # Check for keywords in the message
        matched_categories = set()
        for category, terms in keywords.items():
            for term in terms:
                if term in message_lower:
                    matched_categories.add(category)
        
        # Extract relevant information from knowledge base
        for kb_name, kb_data in self.knowledge_base.items():
            for section_name, section_data in kb_data.items():
                # Check if section is relevant to matched categories
                if any(category in section_name.lower() for category in matched_categories) or \
                   any(category in kb_name.lower() for category in matched_categories):
                    # Convert section data to string format
                    if isinstance(section_data, dict):
                        section_text = f"Информация о {section_name}:\n"
                        for key, value in section_data.items():
                            if isinstance(value, list):
                                section_text += f"- {key}: {', '.join(value)}\n"
                            else:
                                section_text += f"- {key}: {value}\n"
                    elif isinstance(section_data, list):
                        section_text = f"Информация о {section_name}:\n"
                        for item in section_data:
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    section_text += f"- {k}: {v}\n"
                            else:
                                section_text += f"- {item}\n"
                    else:
                        section_text = f"Информация о {section_name}: {section_data}\n"
                    
                    context_parts.append(section_text)
        
        # Join all relevant information
        context = "\n".join(context_parts)
        
        return context
    
    def _generate_openai_response(self, message, session_id):
        """Generate a response using OpenAI.
        
        Args:
            message: The user's message.
            session_id: The session ID.
            
        Returns:
            The generated response.
        """
        # Check if OpenAI is properly configured
        if not openai or not self.openai_api_key:
            # Fall back to demo response if OpenAI is not available
            logger.warning("OpenAI not properly configured. Using demo response.")
            return self._generate_demo_response(message)
        
        try:
            session = self._get_session(session_id)
            
            # Get relevant knowledge base context
            knowledge_context = self._generate_knowledge_context(message)
            
            # Prepare conversation history
            messages = []
            if session and session["messages"]:
                # Add up to 10 most recent messages for context
                for msg in session["messages"][-10:]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add system message with detailed instructions and knowledge base context
            system_prompt = """Ты AI-ассистент проекта Lucky Train на блокчейне TON. Твоя задача - давать подробные и исчерпывающие ответы на вопросы о проекте, токене LTT, метавселенной и блокчейне."""
            
            if knowledge_context:
                system_prompt += f"\n\nИспользуй следующую информацию из базы знаний:\n{knowledge_context}"
            
            messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": message
            })
            
            # Generate response
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Extract response
            response = completion.choices[0].message.content
            
            return response
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            # Fall back to demo response in case of error
            return self._generate_demo_response(message)
    
    def _generate_demo_response(self, message):
        """Generate a demo response without using OpenAI.
        
        Args:
            message: The user's message.
            
        Returns:
            The demo response.
        """
        # Get knowledge context if available
        knowledge_context = self._generate_knowledge_context(message)
        
        # Predefined demo responses for common questions
        message_lower = message.lower()
        
        if "привет" in message_lower or "здравствуй" in message_lower or "хай" in message_lower:
            return "Привет! Я AI-ассистент проекта Lucky Train. Я работаю в демо-режиме, так как не настроен ключ API OpenAI. Как я могу помочь вам сегодня?"
        
        if "что такое lucky train" in message_lower or "расскажи о проекте" in message_lower:
            return """Lucky Train - это инновационный метавселенный проект на блокчейне TON с собственной токеномикой. Проект создает виртуальный мир с игровыми элементами, где пользователи могут взаимодействовать, строить и участвовать в экономике, основанной на токене LTT.

В настоящее время я работаю в демо-режиме без доступа к API OpenAI. Для полной функциональности, пожалуйста, настройте ключ API OpenAI."""
        
        if "токен" in message_lower or "ltt" in message_lower:
            return """Lucky Train Token (LTT) - это основная валюта экосистемы Lucky Train. Это дефляционный токен, который используется для:
- Транзакций в метавселенной
- Стейкинга
- Участия в управлении проектом

В настоящее время я работаю в демо-режиме без доступа к API OpenAI. Для получения более подробной информации, пожалуйста, настройте ключ API OpenAI."""
        
        if "блокчейн" in message_lower or "ton" in message_lower:
            return """Lucky Train построен на блокчейне TON (The Open Network). TON обеспечивает быстрые транзакции, низкие комиссии и высокую масштабируемость, что делает его идеальным выбором для метавселенных проектов.

В настоящее время я работаю в демо-режиме без доступа к API OpenAI. Для получения более подробной информации, пожалуйста, настройте ключ API OpenAI."""
        
        if "метавселенн" in message_lower:
            return """Метавселенная Lucky Train - это виртуальный мир с игровыми элементами, где пользователи могут взаимодействовать, строить и участвовать в экономике на основе токена LTT. Она сочетает в себе социальные, игровые и финансовые элементы в единой экосистеме.

В настоящее время я работаю в демо-режиме без доступа к API OpenAI. Для получения более подробной информации, пожалуйста, настройте ключ API OpenAI."""
        
        if "команд" in message_lower or "разработчик" in message_lower:
            return """Команда Lucky Train состоит из опытных специалистов в области блокчейна, игровой индустрии и финансов. 

В настоящее время я работаю в демо-режиме без доступа к API OpenAI. Для получения более подробной информации о команде, пожалуйста, настройте ключ API OpenAI."""
        
        if "api" in message_lower or "openai" in message_lower or "ключ" in message_lower or "настрой" in message_lower:
            return """Для настройки ключа API OpenAI и активации полной функциональности AI-ассистента, выполните следующие шаги:

1. Получите ключ API на сайте OpenAI (https://platform.openai.com)
2. Создайте файл .env в корневой директории проекта
3. Добавьте строку: OPENAI_API_KEY=ваш_ключ_api_openai
4. Перезапустите приложение

После этого AI-ассистент будет использовать API OpenAI для генерации более точных и подробных ответов."""
        
        # Use knowledge context if available
        if knowledge_context:
            return f"""Вот информация из базы знаний по вашему запросу:

{knowledge_context}

Обратите внимание, я работаю в демо-режиме без доступа к API OpenAI. Для более подробной и точной информации, пожалуйста, настройте ключ API OpenAI."""
        
        # Default response
        return """Я AI-ассистент проекта Lucky Train, но сейчас работаю в демо-режиме, так как не настроен ключ API OpenAI. Я могу дать базовую информацию о проекте, токене LTT и блокчейне TON.

Для активации полной функциональности, пожалуйста, добавьте ключ API OpenAI через файл .env или переменную окружения OPENAI_API_KEY."""
    
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
        
    def serve_favicon(self):
        """Serve the favicon.ico file.
        
        Returns:
            The favicon file.
        """
        return send_from_directory(self.app.static_folder, "images/favicon.ico")
    
    def run(self, host: str = "0.0.0.0", port: int = 10000, debug: bool = False):
        """Run the web interface demo.
        
        Args:
            host: The host to run on.
            port: The port to run on.
            debug: Whether to run in debug mode.
        """
        logger.info(f"Starting web interface demo on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Run the web interface if executed directly
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("WEB_PORT", 10000))
    
    # Initialize the web interface demo
    demo = LuckyTrainWebInterfaceDemo()
    
    # Run the demo
    demo.run(port=port, debug=True) 