"""
Integrated Web Interface Demo for Lucky Train AI Assistant

This module provides a web interface for demonstrating the LuckyTrainAI system using
the unified system architecture that integrates all components.
"""

import os
import logging
from datetime import datetime
import time
import uuid
import json
import dotenv

# Load environment variables from .env file if it exists
try:
    dotenv.load_dotenv()
    logging.info("Loaded environment variables from .env file")
except Exception as e:
    logging.warning(f"Could not load .env file: {e}")

# Import required Flask modules
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory, session
from flask_cors import CORS

# Import unified system integrator
from unified_system_integrator import create_unified_system, UnifiedSystem

# Make requests module optional - we don't actually use it directly in the demo mode
try:
    import requests
except ImportError:
    requests = None
    logging.warning("The 'requests' module is not available. Some features may be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LuckyTrainWebInterfaceDemo:
    """Integrated web interface for the Lucky Train AI assistant."""
    
    def __init__(self, config_path=None, openai_api_key=None, knowledge_base_path=None):
        """Initialize the web interface demo.
        
        Args:
            config_path: Path to the configuration file
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
        self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        
        # Determine if running in demo mode
        self.demo_mode = not self.openai_api_key
        if self.demo_mode:
            logger.info("Running in demo mode - using predefined responses")
        
        # Initialize knowledge base
        self.knowledge_base_path = knowledge_base_path or os.path.join(os.path.dirname(__file__), "knowledge_base")
        self.knowledge_base = self._load_knowledge_base()
        
        # Initialize unified system if not in demo mode
        self.unified_system = None
        if not self.demo_mode:
            try:
                logger.info("Initializing unified system...")
                self.unified_system = create_unified_system(config_path)
                logger.info("Unified system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize unified system: {e}")
                logger.warning("Falling back to demo mode")
                self.demo_mode = True
        
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
        
        # Unified system API routes
        self.app.route("/api/unified", methods=["POST"])(self.unified_api)
        self.app.route("/api/blockchain", methods=["GET", "POST"])(self.blockchain_api)
        self.app.route("/api/metaverse", methods=["GET", "POST"])(self.metaverse_api)
        
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
            title="Lucky Train AI Assistant Demo",
            unified_mode=not self.demo_mode
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
            unified_mode=not self.demo_mode,
            welcome_message="Привет! Я AI-ассистент проекта Lucky Train. Задайте мне вопрос о проекте, и я постараюсь дать вам детальный ответ."
        )
    
    def test_page(self):
        """Handle the test page route.
        
        Returns:
            The rendered test page.
        """
        return render_template(
            "test.html",
            title="Test Page",
            unified_mode=not self.demo_mode
        )
    
    def simple_page(self):
        """Handle the simple page route.
        
        Returns:
            The rendered simple page.
        """
        return render_template(
            "simple.html",
            title="Simple Lucky Train AI Page",
            unified_mode=not self.demo_mode
        )
    
    def luckytrainai_index(self):
        """Handle the LuckyTrainAI index route.
        
        Returns:
            The rendered LuckyTrainAI index page.
        """
        # Get connected services if unified system is active
        connected_services = []
        if self.unified_system:
            connected_services = list(self.unified_system.services.keys())
        
        return render_template(
            "luckytrainai.html",
            title="LuckyTrainAI Demo",
            theme="dark",
            unified_mode=not self.demo_mode,
            connected_services=connected_services,
            welcome_message="Привет! Я AI-ассистент проекта Lucky Train. Задайте мне вопрос о проекте, и я постараюсь дать вам детальный ответ на основе актуальной информации."
        )
    
    def luckytrainai_chat(self):
        """Handle the LuckyTrainAI chat route.
        
        Returns:
            The rendered LuckyTrainAI chat page.
        """
        # Get connected services if unified system is active
        connected_services = []
        if self.unified_system:
            connected_services = list(self.unified_system.services.keys())
            
        return render_template(
            "luckytrainai-chat.html",
            title="LuckyTrainAI Demo - Чат",
            theme="dark",
            unified_mode=not self.demo_mode,
            connected_services=connected_services,
            welcome_message="Привет! Я AI-ассистент проекта Lucky Train. Задайте мне вопрос о проекте, и я постараюсь дать вам детальный ответ на основе актуальной информации."
        )
    
    def settings_api(self):
        """Handle the settings API route.
        
        Returns:
            JSON response with the current settings or updates settings.
        """
        if request.method == "GET":
            # Return current settings (without exposing API key)
            settings = {
                "has_openai_api_key": bool(self.openai_api_key),
                "has_knowledge_base": bool(self.knowledge_base),
                "demo_mode": self.demo_mode,
                "unified_mode": not self.demo_mode
            }
            
            # Add unified system info if available
            if self.unified_system:
                settings["unified_system"] = {
                    "available_services": list(self.unified_system.services.keys()),
                    "active_services": [
                        name for name, thread in self.unified_system.service_threads.items() 
                        if thread.is_alive()
                    ]
                }
                
            return jsonify(settings)
            
        elif request.method == "POST":
            # Update settings
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            # Update OpenAI API key if provided
            api_key = data.get("openai_api_key")
            if api_key and '\0' not in api_key:
                self.openai_api_key = api_key
                os.environ["OPENAI_API_KEY"] = api_key
                
                # Re-initialize unified system if we were in demo mode
                if self.demo_mode and not self.unified_system:
                    try:
                        self.unified_system = create_unified_system()
                        self.demo_mode = False
                        logger.info("Unified system initialized with new API key")
                    except Exception as e:
                        logger.error(f"Failed to initialize unified system: {e}")
                
                logger.info("OpenAI API key updated")
                
                return jsonify({
                    "success": True,
                    "message": "API key updated successfully",
                    "demo_mode": self.demo_mode,
                    "unified_mode": not self.demo_mode
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
            # Check if we should use unified system
            if self.unified_system and "knowledge_base" in self.unified_system.services:
                # Use knowledge base service from unified system
                kb_service = self.unified_system.services["knowledge_base"]
                
                # Return list of knowledge base entries (implementation depends on the actual service)
                if hasattr(kb_service, "list_entries"):
                    kb_list = kb_service.list_entries()
                    return jsonify({"knowledge_base": kb_list})
            
            # Fallback to local knowledge base
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
            
            # Check if we should use unified system
            if self.unified_system and "knowledge_base" in self.unified_system.services:
                # Use knowledge base service from unified system
                kb_service = self.unified_system.services["knowledge_base"]
                
                # Add or update entry (implementation depends on the actual service)
                if hasattr(kb_service, "add_entry"):
                    success = kb_service.add_entry(name, content)
                    if success:
                        return jsonify({
                            "success": True,
                            "message": "Knowledge base updated successfully"
                        })
                    else:
                        return jsonify({
                            "error": "Failed to update knowledge base"
                        }), 500
            
            # Fallback to local knowledge base
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
            
            # Use unified system if available, otherwise use demo response
            if not self.demo_mode and self.unified_system:
                # Prepare request data
                request_data = {
                    "message": message,
                    "user_id": user_id,
                    "platform": "web",
                    "session_id": session_id
                }
                
                # Handle request through unified system
                result = self.unified_system.handle_request(request_data)
                
                # Extract response
                if isinstance(result, dict) and "response" in result:
                    response = result["response"]
                else:
                    response = str(result)
            else:
                # Generate demo response
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
                "response_time": response_time,
                "unified_mode": not self.demo_mode
            })
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    
    def unified_api(self):
        """Handle the unified system API route.
        
        Returns:
            JSON response with the result from the unified system.
        """
        if self.demo_mode or not self.unified_system:
            return jsonify({
                "error": "Unified system not available",
                "demo_mode": self.demo_mode
            }), 400
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        if "action" not in data:
            return jsonify({"error": "Action is required"}), 400
        
        action = data["action"]
        
        try:
            if action == "get_services":
                # Return list of available services
                return jsonify({
                    "services": list(self.unified_system.services.keys()),
                    "active_services": [
                        name for name, thread in self.unified_system.service_threads.items() 
                        if thread.is_alive()
                    ]
                })
                
            elif action == "start_service":
                # Start a service
                service_name = data.get("service")
                if not service_name:
                    return jsonify({"error": "Service name is required"}), 400
                    
                if service_name not in self.unified_system.services:
                    return jsonify({"error": f"Service {service_name} not found"}), 404
                
                # Start the service
                self.unified_system.start_services([service_name])
                
                return jsonify({
                    "success": True,
                    "message": f"Service {service_name} started"
                })
                
            elif action == "stop_service":
                # Stop a service
                service_name = data.get("service")
                if not service_name:
                    return jsonify({"error": "Service name is required"}), 400
                    
                if service_name not in self.unified_system.services:
                    return jsonify({"error": f"Service {service_name} not found"}), 404
                
                # Stop the service
                self.unified_system.stop_services([service_name])
                
                return jsonify({
                    "success": True,
                    "message": f"Service {service_name} stopped"
                })
                
            elif action == "handle_request":
                # Handle a request through the unified system
                request_data = data.get("request_data")
                route = data.get("route")
                
                if not request_data:
                    return jsonify({"error": "Request data is required"}), 400
                
                # Handle the request
                result = self.unified_system.handle_request(request_data, route)
                
                return jsonify({
                    "result": result
                })
                
            else:
                return jsonify({"error": f"Unknown action: {action}"}), 400
                
        except Exception as e:
            logger.error(f"Error handling unified API request: {e}")
            
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    
    def blockchain_api(self):
        """Handle blockchain API requests.
        
        Returns:
            JSON response with blockchain data or operation result.
        """
        if self.demo_mode or not self.unified_system or "blockchain" not in self.unified_system.services:
            # Return demo data in demo mode
            if request.method == "GET":
                return jsonify({
                    "demo": True,
                    "blockchain": "TON",
                    "token": "LTT",
                    "price": "0.0123 TON",
                    "market_cap": "1,230,000 TON",
                    "holders": 3500,
                    "transactions": 12500
                })
            return jsonify({"error": "Blockchain service not available"}), 400
        
        # Get blockchain service
        blockchain_service = self.unified_system.services["blockchain"]
        
        if request.method == "GET":
            # Return blockchain info
            try:
                return jsonify(blockchain_service.get_info())
            except Exception as e:
                logger.error(f"Error getting blockchain info: {e}")
                return jsonify({"error": str(e)}), 500
                
        elif request.method == "POST":
            # Perform blockchain operation
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
                
            operation = data.get("operation")
            if not operation:
                return jsonify({"error": "Operation is required"}), 400
                
            try:
                if operation == "get_balance":
                    address = data.get("address")
                    if not address:
                        return jsonify({"error": "Address is required"}), 400
                        
                    balance = blockchain_service.get_balance(address)
                    return jsonify({"balance": balance})
                    
                elif operation == "get_transaction":
                    tx_id = data.get("tx_id")
                    if not tx_id:
                        return jsonify({"error": "Transaction ID is required"}), 400
                        
                    tx = blockchain_service.get_transaction(tx_id)
                    return jsonify({"transaction": tx})
                    
                else:
                    return jsonify({"error": f"Unknown operation: {operation}"}), 400
                    
            except Exception as e:
                logger.error(f"Error performing blockchain operation: {e}")
                return jsonify({"error": str(e)}), 500
    
    def metaverse_api(self):
        """Handle metaverse API requests.
        
        Returns:
            JSON response with metaverse data or operation result.
        """
        if self.demo_mode or not self.unified_system or "metaverse" not in self.unified_system.services:
            # Return demo data in demo mode
            if request.method == "GET":
                return jsonify({
                    "demo": True,
                    "metaverse": "Lucky Train World",
                    "online_users": 120,
                    "locations": ["Central Station", "Market Square", "Mining District", "Residential Area"],
                    "events": [
                        {"name": "Token Hunt", "time": "Daily at 18:00 UTC"},
                        {"name": "Builder's Challenge", "time": "Weekends"}
                    ]
                })
            return jsonify({"error": "Metaverse service not available"}), 400
        
        # Get metaverse service
        metaverse_service = self.unified_system.services["metaverse"]
        
        if request.method == "GET":
            # Return metaverse info
            try:
                return jsonify(metaverse_service.get_info())
            except Exception as e:
                logger.error(f"Error getting metaverse info: {e}")
                return jsonify({"error": str(e)}), 500
                
        elif request.method == "POST":
            # Perform metaverse operation
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
                
            operation = data.get("operation")
            if not operation:
                return jsonify({"error": "Operation is required"}), 400
                
            try:
                if operation == "get_user_location":
                    user_id = data.get("user_id")
                    if not user_id:
                        return jsonify({"error": "User ID is required"}), 400
                        
                    location = metaverse_service.get_user_location(user_id)
                    return jsonify({"location": location})
                    
                elif operation == "send_notification":
                    user_id = data.get("user_id")
                    message = data.get("message")
                    
                    if not user_id or not message:
                        return jsonify({"error": "User ID and message are required"}), 400
                        
                    success = metaverse_service.send_notification(user_id, message)
                    return jsonify({"success": success})
                    
                else:
                    return jsonify({"error": f"Unknown operation: {operation}"}), 400
                    
            except Exception as e:
                logger.error(f"Error performing metaverse operation: {e}")
                return jsonify({"error": str(e)}), 500
    
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
        # Start unified system services if available
        if self.unified_system:
            try:
                # Start essential services
                essential_services = ["api_gateway", "assistant_core", "knowledge_base", "ai_model"]
                self.unified_system.start_services(essential_services)
                logger.info(f"Started essential unified system services: {essential_services}")
            except Exception as e:
                logger.error(f"Failed to start unified system services: {e}")
        
        logger.info(f"Starting web interface demo on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Run the web interface if executed directly
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Integrated Web Interface Demo")
    parser.add_argument("--port", type=int, default=10000, help="Port to run the web interface on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the web interface on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--demo", action="store_true", help="Force demo mode")
    parser.add_argument("--openai-api-key", help="OpenAI API key")
    parser.add_argument("--knowledge-base-path", help="Path to knowledge base files")
    
    args = parser.parse_args()
    
    # Get port from environment variable (Render sets PORT) or use default
    port = int(os.environ.get("PORT", os.environ.get("WEB_PORT", args.port)))
    
    # Force demo mode if specified
    openai_api_key = args.openai_api_key
    if args.demo:
        openai_api_key = None
    
    # Initialize the web interface demo
    demo = LuckyTrainWebInterfaceDemo(
        config_path=args.config,
        openai_api_key=openai_api_key,
        knowledge_base_path=args.knowledge_base_path
    )
    
    # Run the demo
    demo.run(host=args.host, port=port, debug=args.debug) 