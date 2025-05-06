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
import requests

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
        if openai_api_key and '\0' not in openai_api_key:
            self.openai_api_key = openai_api_key
        elif os.environ.get("OPENAI_API_KEY") and '\0' not in os.environ.get("OPENAI_API_KEY", ""):
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
            
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("OpenAI API key configured successfully")
        else:
            logger.warning("No OpenAI API key provided. AI functionality will be limited to demo responses.")
        
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
        system_prompt = """Ты AI-ассистент проекта Lucky Train на блокчейне TON. Твоя задача - давать подробные и исчерпывающие ответы на вопросы о проекте, токене LTT, метавселенной и блокчейне.

Правила:
1. Давай детальные и содержательные ответы на вопросы пользователей
2. Используй информацию из базы знаний, приведенную ниже
3. Если информации недостаточно, укажи это, но предложи наиболее вероятный ответ
4. Объясняй сложные термины простым языком
5. Отвечай с энтузиазмом и дружелюбием
6. Используй форматирование текста для улучшения читаемости

Информация из базы знаний:
"""
        if knowledge_context:
            system_prompt += knowledge_context
        else:
            system_prompt += """
Проект Lucky Train - это метавселенная на блокчейне TON с собственной токеномикой на основе LTT (Lucky Train Token).
Токен LTT имеет дефляционную модель и используется для транзакций внутри метавселенной, стейкинга и участия в управлении проектом.
Метавселенная позволяет пользователям взаимодействовать, строить и участвовать в экономике проекта через различные активности.
"""
        
        messages.insert(0, {
            "role": "system",
            "content": system_prompt
        })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        try:
            # Call OpenAI API with GPT-4 for better detailed responses
            completion = openai.ChatCompletion.create(
                model="gpt-4",  # Using GPT-4 for more detailed responses
                messages=messages,
                max_tokens=1000,  # Increased token limit for more detailed responses
                temperature=0.7
            )
            
            # Extract response
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            logger.error(f"Error calling OpenAI API with GPT-4: {e}")
            try:
                # Fallback to GPT-4o mini if GPT-4 fails
                completion = openai.ChatCompletion.create(
                    model="gpt-4o-mini",  # Using GPT-4o mini as a fallback option
                    messages=messages,
                    max_tokens=900,
                    temperature=0.7
                )
                
                # Extract response
                response = completion.choices[0].message.content
                logger.info("Successfully generated response using GPT-4o mini")
                return response
            except Exception as e2:
                logger.error(f"Error calling OpenAI API with GPT-4o mini: {e2}")
                try:
                    # Fallback to GPT-3.5 if GPT-4o mini also fails
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        max_tokens=800,
                        temperature=0.7
                    )
                    
                    # Extract response
                    response = completion.choices[0].message.content
                    logger.info("Successfully generated response using GPT-3.5-turbo")
                    return response
                except Exception as e3:
                    logger.error(f"Error calling OpenAI API with fallback model GPT-3.5-turbo: {e3}")
                    # Fallback to demo response if all API calls fail
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
            return "Привет! Рад видеть вас в демо-версии LuckyTrainAI. Я готов предоставить вам детальную информацию о проекте Lucky Train, его токеномике, метавселенной и технологиях. Что именно вас интересует?"
        
        if "блокчейн" in message_lower or "ton" in message_lower:
            return """LuckyTrain построен на блокчейне TON (The Open Network), который предоставляет множество преимуществ для проекта:

1. Высокая скорость транзакций (до 100,000 транзакций в секунду)
2. Низкие комиссии за транзакции (в среднем менее $0.01)
3. Масштабируемая архитектура с поддержкой шардинга
4. Поддержка смарт-контрактов на языке FunC
5. Большое и активно растущее сообщество разработчиков

TON идеально подходит для игровых и метавселенных проектов благодаря своей высокой производительности и низким затратам на транзакции, что позволяет создавать экономические системы с микротранзакциями без чрезмерных комиссий."""
        
        if "метавселенн" in message_lower:
            return """Метавселенная LuckyTrain - это многофункциональная виртуальная платформа, где:

1. Пользователи могут создавать индивидуальные аватары и персонализировать их с помощью NFT-предметов
2. Виртуальное пространство разделено на различные тематические зоны, каждая со своими уникальными функциями и активностями
3. Игроки могут приобретать виртуальные земельные участки (LAND) и строить на них объекты
4. Экономическая система построена на токене LTT, который используется для всех транзакций внутри метавселенной
5. Имеются социальные хабы для взаимодействия пользователей, маркетплейсы для торговли NFT и виртуальными товарами
6. Регулярно проводятся события и активности, поощряющие участие пользователей

Метавселенная интегрирована с технологией блокчейн TON, что обеспечивает прозрачность всех транзакций и подлинное владение цифровыми активами."""
        
        if "токен" in message_lower or "ltt" in message_lower:
            return """LTT (Lucky Train Token) - это основная валюта экосистемы Lucky Train со следующими характеристиками и применениями:

1. Дефляционная модель - часть токенов сжигается при каждой транзакции, что обеспечивает постепенное сокращение предложения и потенциальный рост стоимости
2. Общий запас: 1,000,000,000 LTT
3. Распределение:
   - 40% для экосистемы и вознаграждений пользователей
   - 20% для команды и ранних инвесторов (с периодом вестинга)
   - 15% для публичной продажи
   - 15% для ликвидности на DEX и CEX
   - 10% для стратегического резерва

Применения токена LTT:
- Покупка виртуальных земель и недвижимости в метавселенной
- Торговля NFT и виртуальными товарами на маркетплейсе
- Участие в стейкинге с различными периодами блокировки и вознаграждениями
- Голосование по вопросам управления проектом (DAO)
- Оплата внутриигровых услуг и активностей

Токен LTT построен на стандарте TON и доступен на нескольких децентрализованных и централизованных биржах."""
        
        if "карт" in message_lower:
            return """Дорожная карта проекта Lucky Train включает следующие основные фазы развития:

Фаза 1 (Q1-Q2 2023) - Фундамент:
- Разработка концепции и экономической модели проекта
- Формирование ключевой команды специалистов
- Создание официального сайта и документации
- Разработка смарт-контрактов для токена LTT
- Частный раунд финансирования

Фаза 2 (Q3-Q4 2023) - Запуск токена:
- Публичная продажа токена LTT
- Листинг на децентрализованных биржах
- Реализация стейкинг-программы
- Запуск альфа-версии метавселенной для избранных пользователей
- Маркетинговая кампания для расширения сообщества

Фаза 3 (Q1-Q2 2024) - Развитие экосистемы:
- Открытая бета-версия метавселенной
- Аукцион виртуальных земельных участков
- Запуск NFT-маркетплейса
- Интеграция с внешними Web3-сервисами
- Партнерства с другими проектами на TON

Фаза 4 (Q3-Q4 2024) - Расширение:
- Полный запуск метавселенной
- Внедрение системы управления DAO
- Мобильные приложения для iOS и Android
- Интеграция с VR/AR технологиями
- Листинг на ведущих централизованных биржах

Фаза 5 (2025+) - Масштабирование:
- Расширение возможностей метавселенной
- Кросс-чейн интеграции с другими блокчейнами
- Внедрение AI-элементов в экосистему
- Глобальные партнерства с брендами и проектами
- Реализация реальных применений виртуальных активов

Текущий статус: Проект находится в активной разработке, команда выполняет задачи согласно установленным срокам дорожной карты."""
        
        if "команд" in message_lower:
            return """Команда LuckyTrain состоит из опытных специалистов с обширным опытом в различных областях:

Руководство:
- Алексей Петров - CEO, эксперт в области blockchain с 8-летним опытом, ранее работал в крупных криптопроектах
- Мария Иванова - CTO, более 10 лет в разработке программного обеспечения, специализация на децентрализованных системах
- Дмитрий Сидоров - COO, опыт управления в IT-секторе, более 15 лет работы с технологическими стартапами

Технический отдел:
- Команда из 12 разработчиков с опытом работы на платформе TON
- 4 специалиста по смарт-контрактам с глубоким знанием FunC
- 6 разработчиков 3D и дизайнеров виртуальных миров
- 3 UI/UX дизайнера с опытом создания интерфейсов для Web3-проектов

Бизнес и маркетинг:
- 5 специалистов по маркетингу и коммуникациям
- 3 эксперта по токеномике и экономическим моделям
- 2 юриста, специализирующихся на криптовалютном регулировании

Консультативный совет:
- Эксперты из индустрии блокчейна, метавселенных и игровой индустрии
- Представители венчурных фондов, инвестирующих в Web3-проекты

Все члены команды имеют подтвержденный опыт работы в своих областях и разделяют общее видение будущего метавселенных на блокчейне TON."""
        
        # Default response
        return """Спасибо за ваш вопрос о проекте Lucky Train! 

Как демо-версия, я могу предоставить вам детальную информацию о различных аспектах проекта:

- Блокчейн TON, на котором построен проект
- Токен LTT, его экономическая модель и применения
- Метавселенная LuckyTrain и ее функциональные возможности
- Дорожная карта проекта и текущий статус разработки
- Команда разработчиков и их опыт
- Инвестиционные возможности и токеномика

Пожалуйста, уточните, какой аспект проекта вас интересует больше всего, и я предоставлю исчерпывающую информацию по этой теме."""
    
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
    # Try to load OpenAI API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize and run the web interface
    web_interface = LuckyTrainWebInterfaceDemo(openai_api_key=api_key)
    
    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get("PORT", 10000))
    
    web_interface.run(host="0.0.0.0", port=port, debug=True) 