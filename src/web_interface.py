"""
Web Interface for Lucky Train AI Assistant

This module provides a web interface for interacting with the Lucky Train AI assistant.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import time
import uuid

from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory, redirect, url_for, session, g
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash
import functools
import sys

# Add the src directory to the Python path to import modules
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from bot.assistant import LuckyTrainAssistant
from streaming import StreamingOutput
from analytics import FeedbackCollector
from security import JWTAuthMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LuckyTrainWebInterface:
    """Web interface for the Lucky Train AI assistant."""
    
    def __init__(self, config_path: str = "./config/config.json"):
        """Initialize the web interface.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config_path = config_path
        self.assistant = LuckyTrainAssistant(config_path)
        self.config = self._load_config(config_path)
        
        # Initialize Flask app
        self.app = Flask(__name__, static_folder="web/static", template_folder="web/templates")
        CORS(self.app)  # Enable CORS for all routes
        
        # Set up session storage
        self.sessions = {}
        
        # Set secret key for flask sessions
        self.app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))
        
        # Set up routes
        self._setup_routes()
        
        # Check for streaming support
        self.streaming_enabled = self.config.get("llm_settings", {}).get("streaming", False)
        if self.streaming_enabled:
            self.streaming_output = StreamingOutput()
        
        # Initialize feedback collector
        self.feedback_collector = FeedbackCollector(self.assistant.analytics_manager)
        
        # Get web interface settings
        self.web_settings = self.config.get("web_interface_settings", {})
        
        # Set up API key management
        self.require_api_key = self.config.get("security_settings", {}).get("api_key_required", False)
        self.valid_api_keys = set(self.config.get("security_settings", {}).get("api_keys", []))
        
        # Add default API key from environment if available
        api_key = os.getenv("WEB_API_KEY")
        if api_key:
            self.valid_api_keys.add(api_key)
        
        # Set up admin credentials
        self.admin_users = self.config.get("admin_settings", {}).get("admin_users", {})
        # Add default admin from environment if available
        admin_username = os.getenv("ADMIN_USERNAME")
        admin_password = os.getenv("ADMIN_PASSWORD")
        if admin_username and admin_password:
            self.admin_users[admin_username] = {
                "password_hash": generate_password_hash(admin_password),
                "role": "admin"
            }
        
        # Initialize JWT auth middleware for secured API routes
        self._setup_jwt_auth()
        
        # Set up secured routes after JWT middleware initialization
        self._setup_secured_routes()
        
        logger.info("Web interface initialized successfully")
    
    def _setup_jwt_auth(self):
        """Set up JWT authentication middleware."""
        # Import security manager from system
        from system_init import get_system
        system = get_system()
        if system and hasattr(system, 'security_manager'):
            self.security_manager = system.security_manager
            
            # Define routes that should be exempt from JWT authentication
            exempt_routes = [
                "/", "/chat", "/docs",
                "/luckytrainai", "/luckytrainai/chat",
                "/api/token", "/api/chat", "/api/stream-chat", 
                "/api/feedback", "/api/analytics", "/api/export-analytics",
                "/admin", "/admin/login", "/admin/logout",
                "/static/<path:path>", "/media/<path:path>"
            ]
            
            # Define URL prefixes that should be exempt from JWT authentication
            exempt_prefixes = [
                "/static/", "/media/", "/admin/"
            ]
            
            # Initialize JWT middleware for v1 API routes
            jwt_middleware = JWTAuthMiddleware(
                app=None,  # We'll initialize it with a Flask blueprint
                security_manager=self.security_manager,
                exempt_routes=exempt_routes,
                exempt_prefixes=exempt_prefixes
            )
            
            # Create a blueprint for secured API routes
            from flask import Blueprint
            secured_api = Blueprint('secured_api', __name__, url_prefix='/api/v1')
            
            # Apply JWT middleware to the blueprint
            jwt_middleware.init_app(secured_api)
            
            # Register the blueprint with the app
            self.app.register_blueprint(secured_api)
            
            # Store the blueprint for later use
            self.secured_api = secured_api
            
            logger.info("JWT authentication middleware configured")
        else:
            logger.warning("Could not initialize JWT middleware: security_manager not available")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load the configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            The configuration as a dictionary.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            # Return default configuration
            return {}
    
    def _setup_routes(self):
        """Set up the routes for the web interface."""
        # Main routes
        self.app.route("/")(self.index)
        self.app.route("/chat")(self.chat_page)
        self.app.route("/docs")(self.docs_page)
        
        # New LuckyTrainAI UI routes
        self.app.route("/luckytrainai")(self.luckytrainai_index)
        self.app.route("/luckytrainai/chat")(self.luckytrainai_chat)
        
        # API routes with improved security
        self.app.route("/api/token", methods=["POST"])(self.get_api_token)
        self.app.route("/api/chat", methods=["POST"])(self.chat_api)
        self.app.route("/api/stream-chat", methods=["POST"])(self.stream_chat_api)
        self.app.route("/api/feedback", methods=["POST"])(self.feedback_api)
        self.app.route("/api/analytics", methods=["GET"])(self.analytics_api)
        self.app.route("/api/export-analytics", methods=["GET"])(self.export_analytics_api)
        
        # Admin routes
        self.app.route("/admin/login", methods=["GET", "POST"])(self.admin_login)
        self.app.route("/admin/logout")(self.admin_logout)
        self.app.route("/admin")(self.admin_required(self.admin_dashboard))
        self.app.route("/admin/users")(self.admin_required(self.admin_users_page))
        self.app.route("/admin/api-keys")(self.admin_required(self.admin_api_keys))
        self.app.route("/admin/settings")(self.admin_required(self.admin_settings))
        self.app.route("/admin/knowledge")(self.admin_required(self.admin_knowledge))
        self.app.route("/admin/logs")(self.admin_required(self.admin_logs))
        
        # Admin API routes
        self.app.route("/api/admin/users", methods=["GET", "POST", "PUT", "DELETE"])(self.admin_required(self.admin_users_api))
        self.app.route("/api/admin/api-keys", methods=["GET", "POST", "DELETE"])(self.admin_required(self.admin_api_keys_api))
        self.app.route("/api/admin/settings", methods=["GET", "PUT"])(self.admin_required(self.admin_settings_api))
        self.app.route("/api/admin/analytics", methods=["GET"])(self.admin_required(self.admin_analytics_api))
        self.app.route("/api/admin/logs", methods=["GET"])(self.admin_required(self.admin_logs_api))
        
        # Static files and media
        self.app.route("/static/<path:path>")(self.serve_static)
        self.app.route("/media/<path:path>")(self.serve_media)
        
        # We'll set up the secured API routes after middleware initialization
        logger.info("Base routes configured")
        
    def _setup_secured_routes(self):
        """Set up routes that require JWT authentication."""
        if not hasattr(self, 'secured_api'):
            logger.warning("Secured API blueprint not available, skipping secured routes setup")
            return
            
        # Define routes on the secured blueprint - these will automatically be
        # protected by the JWT middleware
        self.secured_api.route("/chat", methods=["POST"])(self.chat_api_v1_blueprint)
        self.secured_api.route("/stream-chat", methods=["POST"])(self.stream_chat_api_v1_blueprint)
        self.secured_api.route("/blockchain/info", methods=["GET"])(self.blockchain_info_api_blueprint)
        self.secured_api.route("/blockchain/token-info", methods=["GET"])(self.token_info_api_blueprint)
        self.secured_api.route("/metaverse/locations", methods=["GET"])(self.metaverse_locations_api_blueprint)
        
        # Legacy routes - keep these for backward compatibility
        # These use the jwt_secured decorator manually
        self.app.route("/api/v1/chat", methods=["POST"])(self.jwt_secured(self.chat_api_v1))
        self.app.route("/api/v1/stream-chat", methods=["POST"])(self.jwt_secured(self.stream_chat_api_v1))
        self.app.route("/api/v1/blockchain/info", methods=["GET"])(self.jwt_secured(self.blockchain_info_api))
        self.app.route("/api/v1/blockchain/token-info", methods=["GET"])(self.jwt_secured(self.token_info_api))
        self.app.route("/api/v1/metaverse/locations", methods=["GET"])(self.jwt_secured(self.metaverse_locations_api))
        
        logger.info("Secured API routes configured")
    
    def _check_api_key(self) -> bool:
        """Check if the API key is valid.
        
        Returns:
            True if the API key is valid, False otherwise.
        """
        if not self.require_api_key:
            return True
        
        api_key = request.headers.get("X-API-Key")
        return api_key in self.valid_api_keys
    
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
    
    def _get_session(self, session_id: str) -> Optional[Dict]:
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
    
    def _cleanup_old_sessions(self, max_age_seconds: int = 3600):
        """Clean up old sessions.
        
        Args:
            max_age_seconds: Maximum age of sessions in seconds.
        """
        current_time = time.time()
        session_ids_to_remove = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_active"] > max_age_seconds:
                session_ids_to_remove.append(session_id)
        
        for session_id in session_ids_to_remove:
            del self.sessions[session_id]
        
        if session_ids_to_remove:
            logger.info(f"Cleaned up {len(session_ids_to_remove)} old sessions")
    
    def index(self):
        """Handle the root route.
        
        Returns:
            The rendered index page.
        """
        return render_template(
            "index.html",
            title=self.web_settings.get("title", "Lucky Train AI Assistant"),
            theme=self.web_settings.get("theme", "light")
        )
    
    def chat_page(self):
        """Handle the chat page route.
        
        Returns:
            The rendered chat page.
        """
        return render_template(
            "chat.html",
            title=self.web_settings.get("title", "Lucky Train AI Assistant"),
            theme=self.web_settings.get("theme", "light"),
            welcome_message=self.web_settings.get("welcome_message", "Привет! Я официальный AI-ассистент проекта Lucky Train. Чем я могу вам помочь?")
        )
    
    def docs_page(self):
        """Handle the docs page route.
        
        Returns:
            The rendered docs page.
        """
        return render_template(
            "docs.html",
            title=self.web_settings.get("title", "Lucky Train AI Assistant Documentation"),
            theme=self.web_settings.get("theme", "light")
        )
    
    def luckytrainai_index(self):
        """Handle the LuckyTrainAI index route.
        
        Returns:
            The rendered LuckyTrainAI index page.
        """
        return render_template(
            "luckytrainai.html",
            title="LuckyTrainAI",
            theme="dark",
            welcome_message=self.web_settings.get("welcome_message", "Привет! Я официальный AI-ассистент проекта Lucky Train. Чем я могу вам помочь?")
        )
    
    def luckytrainai_chat(self):
        """Handle the LuckyTrainAI chat route.
        
        Returns:
            The rendered LuckyTrainAI chat page.
        """
        return render_template(
            "luckytrainai-chat.html",
            title="LuckyTrainAI - Чат",
            theme="dark",
            welcome_message=self.web_settings.get("welcome_message", "Привет! Я официальный AI-ассистент проекта Lucky Train. Чем я могу вам помочь?")
        )
    
    def chat_api(self):
        """Handle the chat API route.
        
        Returns:
            JSON response with the assistant's reply.
        """
        # Check API key if required
        if not self._check_api_key():
            return jsonify({"error": "Invalid API key"}), 401
        
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
            
            # Get the assistant's response
            response = self.assistant.handle_message(message, user_id, "website")
            
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
            
            # Create feedback buttons
            feedback_buttons = self.feedback_collector.create_feedback_buttons(message_id)
            
            return jsonify({
                "session_id": session_id,
                "message_id": message_id,
                "response": response,
                "feedback": feedback_buttons,
                "response_time": response_time
            })
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
            # Track the error
            self.assistant.analytics_manager.track_error(
                error_type="api_chat",
                error_message=str(e),
                user_id=user_id,
                platform="website"
            )
            
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    
    def stream_chat_api(self):
        """Handle the streaming chat API route.
        
        Returns:
            Streaming response with the assistant's reply.
        """
        # Check if streaming is enabled
        if not self.streaming_enabled:
            return jsonify({
                "error": "Streaming is not enabled"
            }), 400
        
        # Check API key if required
        if not self._check_api_key():
            return jsonify({"error": "Invalid API key"}), 401
        
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
        
        # Create message ID for feedback
        message_id = f"web_{session_id}_{int(time.time())}"
        
        # Get session
        session = self._get_session(session_id)
        if session:
            session["messages"].append({
                "role": "user",
                "content": message,
                "timestamp": time.time()
            })
        
        # Find relevant information from the knowledge base
        relevant_info = self.assistant.find_relevant_information(message)
        
        # Determine the topic of the query
        topic = self.assistant._determine_topic(message)
        
        # Extract contexts from relevant information
        contexts = [info["text"] for info in relevant_info]
        context_text = "\n\n".join(contexts)
        
        # Detect language
        language = self.assistant._detect_language(message)
        
        # Create a system prompt with the knowledge context
        system_prompt = f"""Ты - официальный AI-ассистент проекта Lucky Train на блокчейне TON. 
        Твоя задача - предоставлять точную и полезную информацию о проекте, его токене, метавселенной и блокчейне TON.
        
        Используй следующую информацию для ответа:
        
        {context_text}
        
        Тема запроса: {topic}
        
        Говори уверенно и профессионально. Если информации недостаточно, вежливо скажи, что у тебя нет полных данных по этому вопросу.
        Не выдумывай информацию. Отвечай на языке: {self.assistant.supported_languages.get(language, "русском")} используя соответствующие языковые и культурные особенности."""
        
        # Create message structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        # Get configuration
        model = self.assistant.config.get("llm_settings", {}).get("llm_model", "gpt-3.5-turbo")
        temperature = self.assistant.config.get("temperature", 0.7)
        max_tokens = self.assistant.config.get("max_tokens", 500)
        
        # Function to generate the streaming response
        def generate():
            start_time = time.time()
            full_response = ""
            
            try:
                # Use the streaming handler to get chunks
                for chunk in self.assistant.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                ):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                # Final message with feedback options
                feedback_buttons = self.feedback_collector.create_feedback_buttons(message_id)
                end_time = time.time()
                response_time = end_time - start_time
                
                # Update session with the response
                if session:
                    session["messages"].append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": time.time()
                    })
                
                # Track the interaction
                self.assistant._track_interaction(user_id, "website", message, full_response, response_time)
                
                yield f"data: {json.dumps({'content': '', 'end': True, 'message_id': message_id, 'feedback': feedback_buttons})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                
                # Track the error
                self.assistant.analytics_manager.track_error(
                    error_type="streaming",
                    error_message=str(e),
                    user_id=user_id,
                    platform="website"
                )
                
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    
    def feedback_api(self):
        """Handle the feedback API route.
        
        Returns:
            JSON response indicating success or failure.
        """
        # Check API key if required
        if not self._check_api_key():
            return jsonify({"error": "Invalid API key"}), 401
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        message_id = data.get("message_id")
        if not message_id:
            return jsonify({"error": "No message_id provided"}), 400
        
        rating = data.get("rating")
        if rating is None:
            return jsonify({"error": "No rating provided"}), 400
        
        # Get optional data
        comments = data.get("comments")
        user_id = data.get("user_id", "anonymous")
        platform = data.get("platform", "website")
        
        # Process the feedback
        try:
            self.feedback_collector.process_feedback(
                user_id=user_id,
                platform=platform,
                message_id=message_id,
                rating=rating,
                comments=comments
            )
            
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            
            return jsonify({
                "error": "Failed to process feedback",
                "message": str(e)
            }), 500
    
    def analytics_api(self):
        """Handle the analytics API route.
        
        Returns:
            JSON response with analytics data.
        """
        # Check API key if required
        if not self._check_api_key():
            return jsonify({"error": "Invalid API key"}), 401
        
        # Get query parameters
        date = request.args.get("date")
        
        # Get analytics data
        try:
            analytics_data = self.assistant.get_analytics(date)
            
            return jsonify({
                "success": True,
                "data": analytics_data
            })
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    
    def export_analytics_api(self):
        """Handle the export analytics API route.
        
        Returns:
            CSV file download response.
        """
        # Check API key if required
        if not self._check_api_key():
            return jsonify({"error": "Invalid API key"}), 401
        
        # Get query parameters
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        
        # Create a temporary file for the CSV
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            output_file = temp_file.name
        
        # Export analytics to CSV
        try:
            success = self.assistant.analytics_manager.export_events_to_csv(
                output_file=output_file,
                start_date=start_date,
                end_date=end_date
            )
            
            if success:
                # Set up the file name for download
                file_name = f"lucky_train_analytics_{datetime.now().strftime('%Y-%m-%d')}.csv"
                
                # Return the file for download
                return send_from_directory(
                    os.path.dirname(output_file),
                    os.path.basename(output_file),
                    as_attachment=True,
                    download_name=file_name
                )
            else:
                return jsonify({
                    "error": "Failed to export analytics",
                    "message": "No data available for the specified date range"
                }), 404
            
        except Exception as e:
            logger.error(f"Error exporting analytics: {e}")
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
        finally:
            # Clean up the temporary file
            try:
                os.unlink(output_file)
            except:
                pass
    
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
        logger.info(f"Starting web interface on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
    
    def admin_required(self, f):
        """Decorator to require admin authentication."""
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('admin_logged_in'):
                return redirect(url_for('admin_login'))
            return f(*args, **kwargs)
        return decorated_function
    
    def admin_login(self):
        """Handle admin login.
        
        Returns:
            The rendered admin login page or a redirect to the admin dashboard.
        """
        if request.method == "GET":
            if session.get("admin_logged_in"):
                return redirect(url_for("admin_dashboard"))
            return render_template("admin_login.html")
        
        # Handle POST (login attempt)
        username = request.form.get("username")
        password = request.form.get("password")
        
        admin_user = self.admin_users.get(username)
        if admin_user and check_password_hash(admin_user["password_hash"], password):
            session["admin_logged_in"] = True
            session["admin_username"] = username
            session["admin_role"] = admin_user.get("role", "admin")
            return redirect(url_for("admin_dashboard"))
        
        return render_template("admin_login.html", error="Invalid username or password")
    
    def admin_logout(self):
        """Handle admin logout.
        
        Returns:
            Redirect to admin login page.
        """
        session.pop("admin_logged_in", None)
        session.pop("admin_username", None)
        session.pop("admin_role", None)
        return redirect(url_for("admin_login"))
    
    def admin_dashboard(self):
        """Handle the admin dashboard.
        
        Returns:
            The rendered admin dashboard.
        """
        stats = self._get_system_stats()
        
        return render_template(
            "admin.html",
            admin_name=session.get("admin_username", "Admin"),
            theme=self.web_settings.get("theme", "light"),
            stats=stats,
            active_section="dashboard",
            title="Admin Dashboard - Lucky Train AI"
        )
    
    def admin_users_page(self):
        """Handle the admin users page.
        
        Returns:
            The rendered admin users page.
        """
        return render_template(
            "admin.html",
            admin_name=session.get("admin_username", "Admin"),
            theme=self.web_settings.get("theme", "light"),
            active_section="users",
            title="User Management - Lucky Train AI"
        )
    
    def admin_api_keys(self):
        """Handle the admin API keys page.
        
        Returns:
            The rendered admin API keys page.
        """
        return render_template(
            "admin.html",
            admin_name=session.get("admin_username", "Admin"),
            theme=self.web_settings.get("theme", "light"),
            active_section="api-keys",
            title="API Keys - Lucky Train AI"
        )
    
    def admin_settings(self):
        """Handle the admin settings page.
        
        Returns:
            The rendered admin settings page.
        """
        return render_template(
            "admin.html",
            admin_name=session.get("admin_username", "Admin"),
            theme=self.web_settings.get("theme", "light"),
            active_section="settings",
            title="Settings - Lucky Train AI"
        )
    
    def admin_knowledge(self):
        """Handle the admin knowledge base page.
        
        Returns:
            The rendered admin knowledge base page.
        """
        return render_template(
            "admin.html",
            admin_name=session.get("admin_username", "Admin"),
            theme=self.web_settings.get("theme", "light"),
            active_section="knowledge",
            title="Knowledge Base - Lucky Train AI"
        )
    
    def admin_logs(self):
        """Handle the admin logs page.
        
        Returns:
            The rendered admin logs page.
        """
        return render_template(
            "admin.html",
            admin_name=session.get("admin_username", "Admin"),
            theme=self.web_settings.get("theme", "light"),
            active_section="logs",
            title="System Logs - Lucky Train AI"
        )
    
    def admin_users_api(self):
        """Handle the admin users API.
        
        Returns:
            JSON response with the users data or result of the operation.
        """
        if request.method == "GET":
            # Return list of users (excluding password hashes)
            users = []
            for username, user_data in self.admin_users.items():
                user_copy = user_data.copy()
                user_copy.pop("password_hash", None)
                user_copy["username"] = username
                users.append(user_copy)
            
            return jsonify({"users": users})
        
        elif request.method == "POST":
            # Create a new user
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            username = data.get("username")
            password = data.get("password")
            role = data.get("role", "user")
            
            if not username or not password:
                return jsonify({"error": "Username and password are required"}), 400
            
            if username in self.admin_users:
                return jsonify({"error": "User already exists"}), 409
            
            self.admin_users[username] = {
                "password_hash": generate_password_hash(password),
                "role": role
            }
            
            # Save updated config
            self._save_config()
            
            return jsonify({"success": True, "message": f"User {username} created"})
        
        elif request.method == "PUT":
            # Update an existing user
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            username = data.get("username")
            password = data.get("password")
            role = data.get("role")
            
            if not username:
                return jsonify({"error": "Username is required"}), 400
            
            if username not in self.admin_users:
                return jsonify({"error": "User not found"}), 404
            
            # Update the user
            if password:
                self.admin_users[username]["password_hash"] = generate_password_hash(password)
            
            if role:
                self.admin_users[username]["role"] = role
            
            # Save updated config
            self._save_config()
            
            return jsonify({"success": True, "message": f"User {username} updated"})
        
        elif request.method == "DELETE":
            # Delete a user
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            username = data.get("username")
            
            if not username:
                return jsonify({"error": "Username is required"}), 400
            
            if username not in self.admin_users:
                return jsonify({"error": "User not found"}), 404
            
            # Don't allow deletion of the currently logged in user
            if username == session.get("admin_username"):
                return jsonify({"error": "Cannot delete currently logged in user"}), 403
            
            # Delete the user
            del self.admin_users[username]
            
            # Save updated config
            self._save_config()
            
            return jsonify({"success": True, "message": f"User {username} deleted"})
    
    def admin_api_keys_api(self):
        """Handle the admin API keys API.
        
        Returns:
            JSON response with the API keys data or result of the operation.
        """
        if request.method == "GET":
            # Return list of API keys
            return jsonify({"api_keys": list(self.valid_api_keys)})
        
        elif request.method == "POST":
            # Create a new API key
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            api_key = data.get("api_key")
            
            if not api_key:
                # Generate a new API key if not provided
                api_key = str(uuid.uuid4())
            
            if api_key in self.valid_api_keys:
                return jsonify({"error": "API key already exists"}), 409
            
            self.valid_api_keys.add(api_key)
            
            # Update config
            if "security_settings" not in self.config:
                self.config["security_settings"] = {}
            
            self.config["security_settings"]["api_keys"] = list(self.valid_api_keys)
            
            # Save updated config
            self._save_config()
            
            return jsonify({"success": True, "api_key": api_key})
        
        elif request.method == "DELETE":
            # Delete an API key
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            api_key = data.get("api_key")
            
            if not api_key:
                return jsonify({"error": "API key is required"}), 400
            
            if api_key not in self.valid_api_keys:
                return jsonify({"error": "API key not found"}), 404
            
            self.valid_api_keys.remove(api_key)
            
            # Update config
            self.config["security_settings"]["api_keys"] = list(self.valid_api_keys)
            
            # Save updated config
            self._save_config()
            
            return jsonify({"success": True, "message": "API key deleted"})
    
    def admin_settings_api(self):
        """Handle the admin settings API.
        
        Returns:
            JSON response with the settings data or result of the operation.
        """
        if request.method == "GET":
            # Return current settings
            settings = {
                "web_interface_settings": self.web_settings,
                "security_settings": {
                    "api_key_required": self.require_api_key
                },
                "llm_settings": self.config.get("llm_settings", {})
            }
            
            return jsonify(settings)
        
        elif request.method == "PUT":
            # Update settings
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            # Update web interface settings
            if "web_interface_settings" in data:
                self.web_settings.update(data["web_interface_settings"])
                self.config["web_interface_settings"] = self.web_settings
            
            # Update security settings
            if "security_settings" in data:
                if "api_key_required" in data["security_settings"]:
                    self.require_api_key = data["security_settings"]["api_key_required"]
                    if "security_settings" not in self.config:
                        self.config["security_settings"] = {}
                    self.config["security_settings"]["api_key_required"] = self.require_api_key
            
            # Update LLM settings
            if "llm_settings" in data:
                if "llm_settings" not in self.config:
                    self.config["llm_settings"] = {}
                self.config["llm_settings"].update(data["llm_settings"])
                
                # Update streaming flag if it was changed
                if "streaming" in data["llm_settings"]:
                    self.streaming_enabled = data["llm_settings"]["streaming"]
            
            # Save updated config
            self._save_config()
            
            return jsonify({"success": True, "message": "Settings updated"})
    
    def admin_analytics_api(self):
        """Handle the admin analytics API.
        
        Returns:
            JSON response with analytics data.
        """
        # Get query parameters
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        
        # Get analytics data
        try:
            # Get detailed analytics
            analytics_data = self.assistant.analytics_manager.get_detailed_analytics(
                start_date=start_date,
                end_date=end_date
            )
            
            return jsonify({
                "success": True,
                "data": analytics_data
            })
            
        except Exception as e:
            logger.error(f"Error getting detailed analytics: {e}")
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    
    def admin_logs_api(self):
        """Handle the admin logs API.
        
        Returns:
            JSON response with logs data.
        """
        # Get query parameters
        log_type = request.args.get("type", "all")
        limit = request.args.get("limit", 100)
        try:
            limit = int(limit)
        except:
            limit = 100
        
        try:
            # Get logs
            logs = self._get_system_logs(log_type, limit)
            
            return jsonify({
                "success": True,
                "logs": logs
            })
            
        except Exception as e:
            logger.error(f"Error getting logs: {e}")
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    
    def _get_system_stats(self):
        """Get system statistics.
        
        Returns:
            Dictionary with system statistics.
        """
        try:
            # Get usage statistics
            total_requests = self.assistant.analytics_manager.get_total_requests()
            
            # Get user statistics
            unique_users = self.assistant.analytics_manager.get_unique_users_count()
            
            # Get error rate
            error_rate = self.assistant.analytics_manager.get_error_rate()
            
            # Get average response time
            avg_response_time = self.assistant.analytics_manager.get_average_response_time()
            
            # Get platform usage
            platform_usage = self.assistant.analytics_manager.get_platform_usage()
            
            return {
                "total_requests": total_requests,
                "unique_users": unique_users,
                "error_rate": error_rate,
                "avg_response_time": avg_response_time,
                "platform_usage": platform_usage
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def _get_system_logs(self, log_type: str = "all", limit: int = 100):
        """Get system logs.
        
        Args:
            log_type: Type of logs to get (all, error, info, etc.).
            limit: Maximum number of logs to return.
            
        Returns:
            List of logs.
        """
        try:
            # Define log files based on log type
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
            
            log_files = []
            if log_type == "all" or log_type == "error":
                error_log = os.path.join(log_dir, "error.log")
                if os.path.exists(error_log):
                    log_files.append((error_log, "error"))
            
            if log_type == "all" or log_type == "info":
                info_log = os.path.join(log_dir, "info.log")
                if os.path.exists(info_log):
                    log_files.append((info_log, "info"))
            
            # Read logs
            logs = []
            for log_file, log_level in log_files:
                with open(log_file, "r", encoding="utf-8") as f:
                    # Read the last 'limit' lines
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        # Parse log line
                        parts = line.strip().split(" - ", 3)
                        if len(parts) >= 4:
                            timestamp, logger_name, level, message = parts
                            logs.append({
                                "timestamp": timestamp,
                                "logger": logger_name,
                                "level": level,
                                "message": message,
                                "type": log_level
                            })
            
            # Sort logs by timestamp (newest first)
            logs.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Limit the total number of logs
            return logs[:limit]
            
        except Exception as e:
            logger.error(f"Error getting system logs: {e}")
            return []
    
    def _save_config(self):
        """Save the current configuration to the config file."""
        try:
            # Make sure admin_settings are properly structured
            if "admin_settings" not in self.config:
                self.config["admin_settings"] = {}
            
            # Convert admin users data for storage (with password hashes)
            self.config["admin_settings"]["admin_users"] = self.admin_users
            
            # Write to file
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
                
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def jwt_secured(self, f):
        """Decorator for routes that require JWT authentication.
        
        Args:
            f: Function to wrap
            
        Returns:
            Wrapped function
        """
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Import security manager if not already available
            if not hasattr(self, 'security_manager'):
                from system_init import get_system
                system = get_system()
                self.security_manager = system.security_manager
            
            # Get token from Authorization header
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                return jsonify({"error": "Authorization header is missing"}), 401
            
            # Extract token from Authorization: Bearer <token>
            parts = auth_header.split()
            if parts[0].lower() != 'bearer' or len(parts) != 2:
                return jsonify({"error": "Invalid Authorization header format"}), 401
            
            token = parts[1]
            
            # Verify token
            result = self.security_manager.verify_jwt_token(token)
            
            if not result["valid"]:
                return jsonify({"error": result["error"]}), 401
            
            # Add token payload to kwargs
            kwargs['token_payload'] = result["payload"]
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    def get_api_token(self):
        """Handle the API token request route.
        
        Returns:
            JSON response with the API token
        """
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        api_key = data.get("api_key")
        if not api_key:
            return jsonify({"error": "No API key provided"}), 400
        
        # Check if API key is valid
        if api_key not in self.valid_api_keys:
            return jsonify({"error": "Invalid API key"}), 401
        
        # Get user info
        user_id = data.get("user_id", f"api_user_{api_key[:8]}")
        role = data.get("role", "api_user")
        
        # Import security manager if not already available
        if not hasattr(self, 'security_manager'):
            from system_init import get_system
            system = get_system()
            self.security_manager = system.security_manager
        
        # Generate JWT token
        token = self.security_manager.generate_jwt_token(user_id, role)
        
        # Return token
        return jsonify({
            "token": token,
            "user_id": user_id,
            "role": role,
            "expires_in": self.security_manager.jwt_expiry_hours * 3600  # seconds
        })
    
    def chat_api_v1(self, token_payload):
        """Handle the secured chat API v1 route.
        
        Args:
            token_payload: JWT token payload
            
        Returns:
            JSON response with the assistant's reply
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
        
        # Get user info from token
        user_id = token_payload.get("sub", f"api_user_{session_id}")
        
        # Handle the message
        try:
            start_time = time.time()
            
            # Get the assistant's response
            response = self.assistant.handle_message(message, user_id, "api")
            
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
            message_id = f"api_{session_id}_{int(time.time())}"
            
            return jsonify({
                "session_id": session_id,
                "message_id": message_id,
                "response": response,
                "response_time": response_time
            })
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
            # Track the error
            self.assistant.analytics_manager.track_error(
                error_type="api_chat_v1",
                error_message=str(e),
                user_id=user_id,
                platform="api"
            )
            
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    
    def stream_chat_api_v1(self, token_payload):
        """Handle the secured streaming chat API v1 route.
        
        Args:
            token_payload: JWT token payload
            
        Returns:
            Streaming response with the assistant's reply
        """
        # Check if streaming is enabled
        if not self.streaming_enabled:
            return jsonify({
                "error": "Streaming is not enabled"
            }), 400
        
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
        
        # Get user info from token
        user_id = token_payload.get("sub", f"api_user_{session_id}")
        
        # Create message ID for feedback
        message_id = f"api_{session_id}_{int(time.time())}"
        
        # Get session
        session = self._get_session(session_id)
        if session:
            session["messages"].append({
                "role": "user",
                "content": message,
                "timestamp": time.time()
            })
        
        # Find relevant information from the knowledge base
        relevant_info = self.assistant.find_relevant_information(message)
        
        # Determine the topic of the query
        topic = self.assistant._determine_topic(message)
        
        # Extract contexts from relevant information
        contexts = [info["text"] for info in relevant_info]
        context_text = "\n\n".join(contexts)
        
        # Detect language
        language = self.assistant._detect_language(message)
        
        # Create a system prompt with the knowledge context
        system_prompt = f"""Ты - официальный AI-ассистент проекта Lucky Train на блокчейне TON. 
        Твоя задача - предоставлять точную и полезную информацию о проекте, его токене, метавселенной и блокчейне TON.
        
        Используй следующую информацию для ответа:
        
        {context_text}
        
        Тема запроса: {topic}
        
        Говори уверенно и профессионально. Если информации недостаточно, вежливо скажи, что у тебя нет полных данных по этому вопросу.
        Не выдумывай информацию. Отвечай на языке: {self.assistant.supported_languages.get(language, "русском")} используя соответствующие языковые и культурные особенности."""
        
        # Create message structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        # Get configuration
        model = self.assistant.config.get("llm_settings", {}).get("llm_model", "gpt-3.5-turbo")
        temperature = self.assistant.config.get("temperature", 0.7)
        max_tokens = self.assistant.config.get("max_tokens", 500)
        
        # Function to generate the streaming response
        def generate():
            start_time = time.time()
            full_response = ""
            
            try:
                # Use the streaming handler to get chunks
                for chunk in self.assistant.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                ):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                # Final message
                end_time = time.time()
                response_time = end_time - start_time
                
                # Update session with the response
                if session:
                    session["messages"].append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": time.time()
                    })
                
                # Track the interaction
                self.assistant._track_interaction(user_id, "api", message, full_response, response_time)
                
                yield f"data: {json.dumps({'content': '', 'end': True, 'message_id': message_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                
                # Track the error
                self.assistant.analytics_manager.track_error(
                    error_type="streaming_v1",
                    error_message=str(e),
                    user_id=user_id,
                    platform="api"
                )
                
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    
    def blockchain_info_api(self, token_payload):
        """Handle the blockchain info API route.
        
        Args:
            token_payload: JWT token payload
            
        Returns:
            JSON response with blockchain information
        """
        # Import blockchain integration
        from blockchain_integration import TONBlockchainIntegration
        
        # Get blockchain info
        blockchain = TONBlockchainIntegration(self.config_path)
        info = blockchain.get_blockchain_info()
        
        return jsonify(info)
    
    def token_info_api(self, token_payload):
        """Handle the token info API route.
        
        Args:
            token_payload: JWT token payload
            
        Returns:
            JSON response with token information
        """
        # Import blockchain integration
        from blockchain_integration import TONBlockchainIntegration
        
        # Get token symbol from request
        token = request.args.get("token", "LTT")
        
        # Get token info
        blockchain = TONBlockchainIntegration(self.config_path)
        info = blockchain.get_market_price(token)
        
        return jsonify(info)
    
    def metaverse_locations_api(self, token_payload):
        """Handle the metaverse locations API route.
        
        Args:
            token_payload: JWT token payload
            
        Returns:
            JSON response with metaverse locations
        """
        # Load metaverse knowledge base
        try:
            with open(os.path.join(self.assistant.config.get("knowledge_base_path", "./knowledge_base"), "metaverse.json"), "r", encoding="utf-8") as f:
                metaverse_data = json.load(f)
                
            # Extract locations
            locations = metaverse_data.get("locations", [])
            
            return jsonify({"locations": locations})
        except Exception as e:
            logger.error(f"Error loading metaverse locations: {e}")
            return jsonify({"error": "Failed to load metaverse locations"}), 500

    # Blueprint versions of the secure endpoints (using middleware)
    
    def chat_api_v1_blueprint(self):
        """Handle the secured chat API v1 route (blueprint version).
        
        Returns:
            JSON response with the assistant's reply
        """
        # Get the token payload from Flask's g object (set by middleware)
        token_payload = g.token_payload
        
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
        
        # Get user info from token
        user_id = token_payload.get("sub", f"api_user_{session_id}")
        
        # Handle the message
        try:
            start_time = time.time()
            
            # Get the assistant's response
            response = self.assistant.handle_message(message, user_id, "api")
            
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
            message_id = f"api_{session_id}_{int(time.time())}"
            
            return jsonify({
                "session_id": session_id,
                "message_id": message_id,
                "response": response,
                "response_time": response_time
            })
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
            # Track the error
            self.assistant.analytics_manager.track_error(
                error_type="api_chat_v1",
                error_message=str(e),
                user_id=user_id,
                platform="api"
            )
            
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500
    
    def stream_chat_api_v1_blueprint(self):
        """Handle the secured streaming chat API v1 route (blueprint version).
        
        Returns:
            Streaming response with the assistant's reply
        """
        # Get the token payload from Flask's g object (set by middleware)
        token_payload = g.token_payload
        
        # Check if streaming is enabled
        if not self.streaming_enabled:
            return jsonify({
                "error": "Streaming is not enabled"
            }), 400
        
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
        
        # Get user info from token
        user_id = token_payload.get("sub", f"api_user_{session_id}")
        
        # Create message ID for feedback
        message_id = f"api_{session_id}_{int(time.time())}"
        
        # Get session
        session = self._get_session(session_id)
        if session:
            session["messages"].append({
                "role": "user",
                "content": message,
                "timestamp": time.time()
            })
        
        # Find relevant information from the knowledge base
        relevant_info = self.assistant.find_relevant_information(message)
        
        # Determine the topic of the query
        topic = self.assistant._determine_topic(message)
        
        # Extract contexts from relevant information
        contexts = [info["text"] for info in relevant_info]
        context_text = "\n\n".join(contexts)
        
        # Detect language
        language = self.assistant._detect_language(message)
        
        # Create a system prompt with the knowledge context
        system_prompt = f"""Ты - официальный AI-ассистент проекта Lucky Train на блокчейне TON. 
        Твоя задача - предоставлять точную и полезную информацию о проекте, его токене, метавселенной и блокчейне TON.
        
        Используй следующую информацию для ответа:
        
        {context_text}
        
        Тема запроса: {topic}
        
        Говори уверенно и профессионально. Если информации недостаточно, вежливо скажи, что у тебя нет полных данных по этому вопросу.
        Не выдумывай информацию. Отвечай на языке: {self.assistant.supported_languages.get(language, "русском")} используя соответствующие языковые и культурные особенности."""
        
        # Create message structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        # Get configuration
        model = self.assistant.config.get("llm_settings", {}).get("llm_model", "gpt-3.5-turbo")
        temperature = self.assistant.config.get("temperature", 0.7)
        max_tokens = self.assistant.config.get("max_tokens", 500)
        
        # Function to generate the streaming response
        def generate():
            start_time = time.time()
            full_response = ""
            
            try:
                # Use the streaming handler to get chunks
                for chunk in self.assistant.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                ):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                # Final message
                end_time = time.time()
                response_time = end_time - start_time
                
                # Update session with the response
                if session:
                    session["messages"].append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": time.time()
                    })
                
                # Track the interaction
                self.assistant._track_interaction(user_id, "api", message, full_response, response_time)
                
                yield f"data: {json.dumps({'content': '', 'end': True, 'message_id': message_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                
                # Track the error
                self.assistant.analytics_manager.track_error(
                    error_type="streaming_v1",
                    error_message=str(e),
                    user_id=user_id,
                    platform="api"
                )
                
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    
    def blockchain_info_api_blueprint(self):
        """Handle the blockchain info API endpoint (blueprint version).
        
        Returns:
            JSON response with blockchain information
        """
        # Import blockchain integration
        from blockchain_integration import TONBlockchainIntegration
        
        # Get blockchain info
        blockchain = TONBlockchainIntegration(self.config_path)
        info = blockchain.get_blockchain_info()
        
        return jsonify(info)
    
    def token_info_api_blueprint(self):
        """Handle the token info API endpoint (blueprint version).
        
        Returns:
            JSON response with token information
        """
        # Import blockchain integration
        from blockchain_integration import TONBlockchainIntegration
        
        # Get token symbol from request
        token = request.args.get("token", "LTT")
        
        # Get token info
        blockchain = TONBlockchainIntegration(self.config_path)
        info = blockchain.get_market_price(token)
        
        return jsonify(info)
    
    def metaverse_locations_api_blueprint(self):
        """Handle the metaverse locations API endpoint (blueprint version).
        
        Returns:
            JSON response with metaverse locations
        """
        # Load metaverse knowledge base
        try:
            with open(os.path.join(self.assistant.config.get("knowledge_base_path", "./knowledge_base"), "metaverse.json"), "r", encoding="utf-8") as f:
                metaverse_data = json.load(f)
                
            # Extract locations
            locations = metaverse_data.get("locations", [])
            
            return jsonify({"locations": locations})
        except Exception as e:
            logger.error(f"Error loading metaverse locations: {e}")
            return jsonify({"error": "Failed to load metaverse locations"}), 500

# Example usage
if __name__ == "__main__":
    interface = LuckyTrainWebInterface()
    interface.run(port=5000) 