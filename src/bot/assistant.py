"""
Lucky Train AI Assistant

This AI assistant is designed to provide information and support for the Lucky Train project 
on the TON blockchain. It integrates with Telegram and can be deployed on websites and in the metaverse.

The assistant specializes in answering questions about:
- Lucky Train project details
- TON blockchain technology
- Tokenomics
- Project roadmap
- Metaverse integration
- And other related topics
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Union, Generator
import glob
import numpy as np
import openai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import langdetect
import time
import sys

# Add the src directory to the Python path to import modules
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from vector_db import VectorDBHandler
from caching import CacheManager, cached
from analytics import AnalyticsManager
from ai_system import AISystem  # Import the new AI system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LuckyTrainAssistant:
    """Main class for the Lucky Train AI assistant."""
    
    def __init__(self, config_path: str = "../config/config.json"):
        """Initialize the Lucky Train AI assistant.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.knowledge_base = self._load_knowledge_base()
        
        # Initialize the AI system
        try:
            self.ai_system = AISystem(config_path)
            self.use_ai_system = True
            logger.info("AI System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI System: {e}")
            self.use_ai_system = False
        
        # Initialize OpenAI client (legacy/fallback)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            self.use_llm = True
            logger.info("OpenAI client initialized successfully")
        else:
            logger.warning("OpenAI API key not found, falling back to TF-IDF")
            self.use_llm = False
            
        # Initialize vector database handler
        try:
            self.vector_db = VectorDBHandler(config_path)
            self.use_vector_db = True
            # Load knowledge base into vector database if it's empty
            self._ensure_knowledge_base_loaded()
            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            self.use_vector_db = False
            
        # Initialize TF-IDF as fallback
        self.vectorizer = TfidfVectorizer()
        self.kb_vectors = None
        self.kb_texts = []
        self.kb_sources = []
        self._prepare_vectors()
        
        # Initialize caching
        self.cache_manager = CacheManager(self.config.get("caching_settings", {}))
        logger.info("Cache manager initialized")
        
        # Initialize analytics
        self.analytics_manager = AnalyticsManager(self.config.get("analytics_settings", {}))
        logger.info("Analytics manager initialized")
        
        # Load response templates
        self.response_templates = self.config.get("response_templates", {})
        
        # Load supported languages
        self.supported_languages = self.config.get("supported_languages", {
            "ru": "Russian",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi"
        })
        
        # Initialize conversation history storage
        self.conversation_history = {}
        
        # Check for streaming support
        self.streaming_enabled = self.config.get("llm_settings", {}).get("streaming", False)
        
        logger.info("Lucky Train AI Assistant initialized successfully.")
    
    def _ensure_knowledge_base_loaded(self):
        """Ensure knowledge base is loaded into the vector database."""
        if not self.use_vector_db:
            return
        
        # Check if vector database is empty by making a simple search
        test_results = self.vector_db.search("lucky train", top_k=1)
        
        if not test_results:
            # Load knowledge base into vector database
            success = self.vector_db.load_knowledge_base(self.knowledge_base)
            if success:
                logger.info("Successfully loaded knowledge base into vector database")
            else:
                logger.error("Failed to load knowledge base into vector database")
                self.use_vector_db = False
    
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
            return {
                "language": "ru",
                "max_tokens": 1000,
                "temperature": 0.7,
                "knowledge_base_path": "../knowledge_base",
                "supported_platforms": ["telegram", "website", "metaverse"],
                "llm_model": "gpt-3.5-turbo"
            }
    
    def _load_knowledge_base(self) -> Dict:
        """Load the knowledge base from JSON files.
        
        Returns:
            The combined knowledge base as a dictionary.
        """
        knowledge_base = {}
        kb_path = self.config.get("knowledge_base_path", "../knowledge_base")
        
        try:
            # Get all JSON files in the knowledge base directory
            json_files = glob.glob(os.path.join(kb_path, "*.json"))
            
            for file_path in json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Use the filename without extension as the key
                        key = os.path.basename(file_path).split('.')[0]
                        knowledge_base[key] = data
                        logger.info(f"Loaded knowledge base file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load knowledge base file {file_path}: {e}")
            
            return knowledge_base
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return {}
    
    def _prepare_vectors(self):
        """Prepare text chunks for retrieval and vector search."""
        # Flatten the knowledge base into text chunks for vector representation
        texts = []
        sources = []
        
        def process_value(value, path=[]):
            """Recursively process values in the knowledge base."""
            if isinstance(value, dict):
                for k, v in value.items():
                    process_value(v, path + [k])
            elif isinstance(value, list):
                # Handle lists specially - if it's a list of strings, join them
                if all(isinstance(item, str) for item in value):
                    text = ". ".join(value)
                    texts.append(text)
                    sources.append({"path": path, "text": text})
                else:
                    # Otherwise process each item in the list
                    for i, item in enumerate(value):
                        process_value(item, path + [f"item_{i}"])
            elif isinstance(value, str) and len(value.split()) > 3:  # Only include if more than 3 words
                texts.append(value)
                sources.append({"path": path, "text": value})
        
        # Process each knowledge base file
        for kb_name, kb_data in self.knowledge_base.items():
            process_value(kb_data, [kb_name])
        
        if texts:
            self.kb_texts = texts
            self.kb_sources = sources
            
            # Create TF-IDF vectors if not using LLM
            if not self.use_llm:
                self.kb_vectors = self.vectorizer.fit_transform(texts)
                
            logger.info(f"Prepared {len(texts)} text chunks for knowledge retrieval")
        else:
            logger.warning("No text chunks found in knowledge base for retrieval")
    
    @cached(None, key_fn=lambda self, query, top_n=5: f"find_relevant_info:{query}:{top_n}")
    def find_relevant_information(self, query: str, top_n: int = 5) -> List[Dict]:
        """Find the most relevant information for a query using retrieval.
        
        Args:
            query: The user's query.
            top_n: Number of top results to return.
            
        Returns:
            List of dictionaries containing relevant text chunks and their sources.
        """
        # Set the cache manager for the decorator
        find_relevant_information.decorator.manager = self.cache_manager
        
        if not self.kb_texts:
            logger.warning("Knowledge base texts not prepared")
            return []
        
        # If using vector database for embedding search
        if self.use_vector_db:
            try:
                # Use vector database for semantic search
                results = self.vector_db.search(query, top_k=top_n)
                return results
            except Exception as e:
                logger.error(f"Error in vector database search: {e}")
                # Fall back to TF-IDF if there's an error
                if not hasattr(self, 'vectorizer'):
                    self.vectorizer = TfidfVectorizer()
                    self.kb_vectors = self.vectorizer.fit_transform(self.kb_texts)
                
        # If using LLM for embedding search
        if self.use_llm:
            try:
                # Use keyword matching as a simple approach for relevant retrieval
                # In a production system, you would use embeddings API for semantic search
                query_terms = query.lower().split()
                scores = []
                
                for text in self.kb_texts:
                    text_lower = text.lower()
                    score = sum(1 for term in query_terms if term in text_lower)
                    scores.append(score)
                
                # Get indices of top scores
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
                
                results = []
                for idx in top_indices:
                    if scores[idx] > 0:  # Only include if there's at least one keyword match
                        results.append({
                            "text": self.kb_texts[idx],
                            "source": self.kb_sources[idx],
                            "relevance": scores[idx]
                        })
                
                return results
                
            except Exception as e:
                logger.error(f"Error in LLM-based retrieval: {e}")
                # Fall back to TF-IDF if there's an error
                if not hasattr(self, 'vectorizer'):
                    self.vectorizer = TfidfVectorizer()
                    self.kb_vectors = self.vectorizer.fit_transform(self.kb_texts)
                
        # Use TF-IDF based search as fallback
        if not hasattr(self, 'kb_vectors') or self.kb_vectors is None:
            if not hasattr(self, 'vectorizer'):
                self.vectorizer = TfidfVectorizer()
            self.kb_vectors = self.vectorizer.fit_transform(self.kb_texts)
            
        # Transform the query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between the query and all text chunks
        similarities = cosine_similarity(query_vector, self.kb_vectors).flatten()
        
        # Get the indices of the top_n most similar chunks
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Return the top chunks and their sources
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Only include if similarity is above threshold
                results.append({
                    "text": self.kb_texts[idx],
                    "source": self.kb_sources[idx],
                    "similarity": float(similarities[idx])
                })
        
        return results
    
    def generate_response(self, query: str, language: str = None) -> str:
        """Generate a response to a user query based on the knowledge base.
        
        Args:
            query: The user's query.
            language: The language to use for the response. If None, detect the language from the query.
            
        Returns:
            The assistant's response as a string.
        """
        # Detect language if not specified
        if language is None:
            language = self._detect_language(query)
        
        # Clean and normalize the query
        query = self._normalize_text(query)
        
        # Check for special commands or greetings
        if self._is_greeting(query):
            greeting = self.response_templates.get("greeting", "Привет! Я AI-ассистент проекта Lucky Train. Чем я могу вам помочь?")
            # Translate greeting if needed
            if language != "ru":
                greeting = self._translate_text(greeting, language)
            return greeting
        
        # Find relevant information from the knowledge base
        relevant_info = self.find_relevant_information(query)
        
        if not relevant_info:
            no_info_response = self.response_templates.get("not_understood", 
                "Извините, я не нашел информации по вашему запросу. Пожалуйста, попробуйте сформулировать вопрос иначе или спросите о другом аспекте проекта Lucky Train.")
            # Translate no info response if needed
            if language != "ru":
                no_info_response = self._translate_text(no_info_response, language)
            return no_info_response
        
        # Determine the topic of the query
        topic = self._determine_topic(query)
        
        # If using AI system
        if self.use_ai_system:
            # Get current model type from configuration or use default
            model_type = self.config.get("current_ai_model", "ani")
            
            # Generate response using AI system
            response_data = self.ai_system.generate_response(
                query=query,
                context=relevant_info,
                model_type=model_type,
                language=language,
                topic=topic
            )
            
            # Extract response text
            if response_data.get("success", False):
                response = response_data.get("response", "")
            else:
                # Fall back to LLM if AI system fails
                logger.warning("AI system response generation failed, falling back to LLM")
                if self.use_llm:
                    response = self._generate_llm_response(query, relevant_info, topic, language)
                else:
                    # Fallback to traditional response construction
                    response = self._construct_response(query, relevant_info, topic)
                    # Translate the constructed response if needed
                    if language != "ru":
                        response = self._translate_text(response, language)
        
        # If not using AI system
        else:
            # If using LLM for response generation
            if self.use_llm:
                response = self._generate_llm_response(query, relevant_info, topic, language)
            else:
                # Fallback to traditional response construction
                response = self._construct_response(query, relevant_info, topic)
                # Translate the constructed response if needed
                if language != "ru":
                    response = self._translate_text(response, language)
        
        return response
    
    def _generate_llm_response(self, query: str, relevant_info: List[Dict], topic: str, language: str) -> str:
        """Generate a response using an LLM with context from retrieved information.
        
        Args:
            query: The user's query.
            relevant_info: List of dictionaries containing relevant text chunks and their sources.
            topic: The determined topic of the query.
            language: The language to use for the response.
            
        Returns:
            The LLM-generated response.
        """
        try:
            # Extract contexts from relevant information
            contexts = [info["text"] for info in relevant_info]
            context_text = "\n\n".join(contexts)
            
            # Create a system prompt with the knowledge context
            system_prompt = f"""Ты - официальный AI-ассистент проекта Lucky Train на блокчейне TON. 
            Твоя задача - предоставлять точную и полезную информацию о проекте, его токене, метавселенной и блокчейне TON.
            
            Используй следующую информацию для ответа:
            
            {context_text}
            
            Тема запроса: {topic}
            
            Говори уверенно и профессионально. Если информации недостаточно, вежливо скажи, что у тебя нет полных данных по этому вопросу.
            Не выдумывай информацию. Отвечай на языке: {self.supported_languages.get(language, "русском")} используя соответствующие языковые и культурные особенности."""
            
            # Call the OpenAI API
            model = self.config.get("llm_settings", {}).get("llm_model", "gpt-3.5-turbo")
            temperature = self.config.get("temperature", 0.7)
            max_tokens = self.config.get("max_tokens", 500)
            
            # Create message structure
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Check if streaming is enabled
            if self.streaming_enabled:
                # Import the streaming module only when needed
                from streaming import StreamingHandler
                
                # Create a streaming handler if it doesn't exist
                if not hasattr(self, 'streaming_handler'):
                    self.streaming_handler = StreamingHandler()
                
                # Generate a response using streaming
                response_text = self.streaming_handler.generate_streaming_response(
                    openai_client=self.openai_client,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return response_text
            
            # Standard non-streaming approach
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract the response text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.warning("Empty response from LLM")
                # Fall back to traditional response construction
                response = self._construct_response(query, relevant_info, topic)
                # Translate the constructed response if needed
                if language != "ru":
                    response = self._translate_text(response, language)
                return response
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Fall back to traditional response construction
            response = self._construct_response(query, relevant_info, topic)
            # Translate the constructed response if needed
            if language != "ru":
                response = self._translate_text(response, language)
            return response
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace, converting to lowercase, etc.
        
        Args:
            text: The text to normalize.
            
        Returns:
            Normalized text.
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def _is_greeting(self, text: str) -> bool:
        """Check if the text is a greeting.
        
        Args:
            text: The text to check.
            
        Returns:
            True if the text is a greeting, False otherwise.
        """
        greetings = ["привет", "здравствуйте", "добрый день", "доброе утро", "добрый вечер", 
                     "приветствую", "хай", "hello", "hi", "hey"]
        
        # Check if any greeting is in the text
        return any(greeting in text for greeting in greetings)
    
    def _determine_topic(self, query: str) -> str:
        """Determine the topic of the query.
        
        Args:
            query: The user's query.
            
        Returns:
            The topic as a string.
        """
        # Define keywords for different topics
        topics = {
            "project": ["lucky train", "проект", "лаки трейн", "о проекте", "что такое", "миссия", "цель"],
            "token": ["токен", "ltt", "токеномика", "токеномика", "монета", "coin", "стоимость", "цена"],
            "blockchain": ["блокчейн", "тон", "ton", "блок", "транзакция", "шардинг", "консенсус"],
            "metaverse": ["метавселенная", "виртуальный мир", "локации", "поезда", "недвижимость", "аватар"],
            "roadmap": ["дорожная карта", "roadmap", "план", "релиз", "запуск", "будущее", "развитие"],
            "team": ["команда", "основатели", "разработчики", "тим", "создатели"],
            "partners": ["партнеры", "сотрудничество", "альянс", "партнерство"],
            "investment": ["инвестиции", "купить", "приобрести", "вложить", "стейкинг", "доходность"]
        }
        
        # Count the number of keywords from each topic in the query
        topic_counts = {topic: 0 for topic in topics}
        
        for topic, keywords in topics.items():
            for keyword in keywords:
                if keyword in query:
                    topic_counts[topic] += 1
        
        # Find the topic with the most keywords
        max_count = max(topic_counts.values())
        if max_count > 0:
            for topic, count in topic_counts.items():
                if count == max_count:
                    return topic
        
        # Default to "general" if no specific topic is found
        return "general"
    
    def _construct_response(self, query: str, relevant_info: List[Dict], topic: str) -> str:
        """Construct a response based on the query, relevant information, and topic.
        
        Args:
            query: The user's query.
            relevant_info: List of dictionaries containing relevant text chunks and their sources.
            topic: The determined topic of the query.
            
        Returns:
            The constructed response as a string.
        """
        # If no relevant information, return a fallback response
        if not relevant_info:
            return self.response_templates.get("not_understood", 
                "Извините, я не нашел информации по вашему запросу. Пожалуйста, попробуйте сформулировать вопрос иначе.")
        
        # Extract the most relevant text chunks
        chunks = [info["text"] for info in relevant_info[:3]]  # Use top 3 most relevant chunks
        
        # Check if the query is about LTT price or token value
        if "цена" in query or "стоимость" in query or "курс" in query or "сколько стоит" in query:
            return "Токен Lucky Train (LTT) находится на стадии подготовки к запуску. Информация о цене и возможностях приобретения будет объявлена после официального запуска токена. Следите за новостями проекта в наших официальных каналах."
        
        # Check if the query is explicitly about buying tokens
        if "купить" in query or "приобрести" in query or "покупка" in query:
            if "токен" in query or "ltt" in query:
                return "Токен Lucky Train (LTT) будет доступен для покупки после официального запуска. Вы сможете приобрести токены на поддерживаемых биржах и DEX на блокчейне TON. Мы обязательно объявим о запуске и листинге в наших официальных каналах."
        
        # Check if the query is about the project schedule or timeline
        if "когда" in query and ("запуск" in query or "релиз" in query or "выход" in query):
            for chunk in chunks:
                if "roadmap" in chunk.lower() or "дорожная карта" in chunk.lower() or "2024" in chunk:
                    return f"Согласно дорожной карте проекта Lucky Train: {chunk}"
        
        # Construct response based on the topic
        if topic == "project":
            introduction = "Lucky Train - это инновационный проект на блокчейне TON, создающий собственную метавселенную с уникальной концепцией путешествий между виртуальными локациями на поездах."
            details = " ".join(chunks)
            return f"{introduction} {details}"
        
        elif topic == "token":
            introduction = "Lucky Train Token (LTT) - это основная валюта экосистемы Lucky Train, работающая на блокчейне TON."
            details = " ".join(chunks)
            return f"{introduction} {details}"
        
        elif topic == "blockchain":
            introduction = "Lucky Train построен на блокчейне TON (The Open Network), который обеспечивает высокую скорость транзакций и низкие комиссии."
            details = " ".join(chunks)
            return f"{introduction} {details}"
        
        elif topic == "metaverse":
            introduction = "Метавселенная Lucky Train - это виртуальный мир с уникальной концепцией путешествий на поездах между различными локациями."
            details = " ".join(chunks)
            return f"{introduction} {details}"
        
        elif topic == "roadmap":
            introduction = "Дорожная карта проекта Lucky Train включает несколько ключевых этапов развития:"
            details = " ".join(chunks)
            return f"{introduction} {details}"
        
        elif topic == "team":
            introduction = "Проект Lucky Train разрабатывается командой опытных специалистов в области блокчейна, игровой индустрии и виртуальных миров."
            details = " ".join(chunks)
            return f"{introduction} {details}"
        
        elif topic == "partners":
            introduction = "Lucky Train сотрудничает с рядом стратегических партнеров для развития проекта:"
            details = " ".join(chunks)
            return f"{introduction} {details}"
        
        elif topic == "investment":
            introduction = "Инвестиции в проект Lucky Train возможны через приобретение токенов LTT и виртуальных активов в метавселенной."
            details = " ".join(chunks)
            return f"{introduction} {details}"
        
        else:
            # For general or undetermined topics, just combine the relevant chunks
            return " ".join(chunks)
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the text.
        
        Args:
            text: The text to detect language for.
            
        Returns:
            The detected language code (ISO 639-1).
        """
        try:
            # Use langdetect to detect the language
            language = langdetect.detect(text)
            
            # Check if the detected language is supported
            if language in self.supported_languages:
                return language
                
            # If not supported, default to English or Russian
            return self.config.get("language", "ru")
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return self.config.get("language", "ru")
    
    @cached(None, key_fn=lambda self, text, language=None: f"translate:{text}:{language}")
    def _translate_text(self, text: str, target_language: str) -> str:
        """Translate text to the target language using AI.
        
        Args:
            text: The text to translate.
            target_language: The target language code (ISO 639-1).
            
        Returns:
            The translated text.
        """
        # Set the cache manager for the decorator
        _translate_text.decorator.manager = self.cache_manager
        
        if not self.use_llm:
            logger.warning("LLM not available for translation, returning original text")
            return text
            
        try:
            source_language = self._detect_language(text)
            
            # Don't translate if already in target language
            if source_language == target_language:
                return text
            
            # Use OpenAI for translation
            prompt = f"Translate the following text from {self.supported_languages.get(source_language, 'unknown language')} to {self.supported_languages.get(target_language, 'English')}. Preserve any formatting, technical terms, and proper nouns: \n\n{text}"
            
            response = self.openai_client.chat.completions.create(
                model=self.config.get("llm_model", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate the text accurately while preserving the meaning, tone, and cultural context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=self.config.get("max_tokens", 1000)
            )
            
            if response.choices and len(response.choices) > 0:
                translated_text = response.choices[0].message.content
                return translated_text
            else:
                logger.warning("Empty response from translation service")
                return text
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return text
    
    def handle_message(self, text: str, user_id: str = None, platform: str = "telegram") -> str:
        """Handle a message from a user.
        
        Args:
            text: The message text.
            user_id: The ID of the user.
            platform: The platform the message is from.
            
        Returns:
            The assistant's response.
        """
        start_time = time.time()
        
        # Check for AI model switch command
        if text.startswith("/model ") and self.use_ai_system:
            model_name = text.replace("/model ", "").strip().lower()
            available_models = self.ai_system.get_available_models()
            
            if model_name in available_models:
                success = self.ai_system.set_ai_model(model_name)
                if success:
                    response = f"Switched to {model_name.upper()} model."
                else:
                    response = f"Failed to switch to {model_name.upper()} model."
            else:
                response = f"Model {model_name} not available. Available models: {', '.join(available_models)}"
            
            end_time = time.time()
            self._track_interaction(user_id, platform, text, response, end_time - start_time)
            return response
        
        # Check for database query command
        if text.startswith("/db ") and self.use_ai_system:
            parts = text.replace("/db ", "").strip().split(" ", 1)
            if len(parts) >= 2:
                db_type = parts[0].lower()
                query = parts[1]
                
                if db_type in self.ai_system.get_available_connectors():
                    # Connect to database if not already connected
                    self.ai_system.connect_to_database(db_type)
                    
                    # Execute query
                    result = self.ai_system.execute_database_query(db_type, query)
                    
                    if result.get("success", False):
                        if "data" in result:
                            data = result["data"]
                            # Format data as a string table
                            if isinstance(data, list) and data:
                                response = f"Results from {db_type}:\n"
                                # Format as table for first 5 rows
                                for i, row in enumerate(data[:5]):
                                    if i == 0:
                                        response += "\n" + " | ".join(str(k) for k in row.keys()) + "\n"
                                        response += "-" * len(response.split("\n")[-1]) + "\n"
                                    response += " | ".join(str(v) for v in row.values()) + "\n"
                                
                                if len(data) > 5:
                                    response += f"\n... and {len(data) - 5} more rows"
                            else:
                                response = f"Results from {db_type}: {data}"
                        elif "affected_rows" in result:
                            response = f"Query executed successfully on {db_type}. Affected rows: {result['affected_rows']}"
                        else:
                            response = f"Query executed successfully on {db_type}."
                    else:
                        response = f"Error executing query on {db_type}: {result.get('error', 'Unknown error')}"
                else:
                    response = f"Database {db_type} not available. Available databases: {', '.join(self.ai_system.get_available_connectors())}"
                
                end_time = time.time()
                self._track_interaction(user_id, platform, text, response, end_time - start_time)
                return response
        
        # Store message in conversation history
        if user_id:
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []
            
            self.conversation_history[user_id].append({
                "role": "user",
                "content": text,
                "timestamp": time.time()
            })
        
        try:
            # Detect language
            language = self._detect_language(text)
            
            # Generate response
            response = self.generate_response(text, language)
            
            # Store response in conversation history
            if user_id:
                self.conversation_history[user_id].append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": time.time()
                })
                
                # Limit conversation history size
                max_history = self.config.get("max_conversation_history", 20)
                if len(self.conversation_history[user_id]) > max_history:
                    self.conversation_history[user_id] = self.conversation_history[user_id][-max_history:]
            
            # Track interaction for analytics
            end_time = time.time()
            self._track_interaction(user_id, platform, text, response, end_time - start_time)
            
            return response
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return "Извините, произошла ошибка при обработке вашего сообщения. Пожалуйста, попробуйте еще раз."
    
    def _track_interaction(self, user_id: str, platform: str, message: str, response: str, response_time: float):
        """Track a user interaction.
        
        Args:
            user_id: The ID of the user.
            platform: The platform where the message was sent.
            message: The user's message.
            response: The assistant's response.
            response_time: The time taken to generate the response, in seconds.
        """
        # Determine the topic of the message
        topic = self._determine_topic(message)
        
        # Track the message
        self.analytics_manager.track_message(
            user_id=user_id,
            platform=platform,
            message=message,
            response=response,
            topic=topic,
            response_time=response_time
        )

    def get_analytics(self, date: str = None) -> Dict:
        """Get analytics data.
        
        Args:
            date: The date to get daily statistics for, in YYYY-MM-DD format.
                If None, get all metrics.
        
        Returns:
            A dictionary with analytics data.
        """
        if date:
            return self.analytics_manager.get_daily_stats(date) or {}
        else:
            return self.analytics_manager.get_metrics()

    def collect_feedback(self, user_id: str, platform: str, message_id: str, rating: int, comments: str = None):
        """Collect feedback from a user.
        
        Args:
            user_id: The ID of the user.
            platform: The platform where the feedback was given.
            message_id: The ID of the message being rated.
            rating: The rating (e.g., 1-5).
            comments: Optional comments from the user.
        """
        self.analytics_manager.track_feedback(
            user_id=user_id,
            platform=platform,
            message_id=message_id,
            rating=rating,
            comments=comments
        )

    def cleanup_resources(self):
        """Clean up resources used by the assistant."""
        # Stop the analytics manager
        if hasattr(self, 'analytics_manager'):
            self.analytics_manager.stop()
        
        # Clean up any other resources
        
        logger.info("Resources cleaned up successfully.")

    def get_conversation_context(self, user_id: str, max_messages: int = 5) -> List[Dict]:
        """Get the conversation context for a user.
        
        Args:
            user_id: The user's ID.
            max_messages: The maximum number of messages to include in the context.
            
        Returns:
            A list of message dictionaries formatted for the OpenAI API.
        """
        if not user_id or user_id not in self.conversation_history:
            return []
        
        # Get the last N messages
        recent_messages = self.conversation_history[user_id][-max_messages:]
        
        # Format messages for the OpenAI API
        formatted_messages = []
        for message in recent_messages:
            formatted_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        
        return formatted_messages

    def stream_response(self, query: str, user_id: str = None, platform: str = "telegram",
                       language: str = None, bot=None, chat_id=None, websocket_session_id=None) -> Union[str, Generator]:
        """Generate a streaming response to a user query.
        
        Args:
            query: The user's query.
            user_id: The user's ID.
            platform: The platform the message is coming from.
            language: The language to use for the response.
            bot: The Telegram bot instance (for Telegram streaming).
            chat_id: The Telegram chat ID (for Telegram streaming).
            websocket_session_id: The WebSocket session ID (for WebSocket streaming).
            
        Returns:
            A response generator or a string if streaming failed.
        """
        if not self.streaming_enabled or not self.use_llm:
            # Fall back to non-streaming response
            return self.handle_message(query, user_id, platform)
        
        # Import the streaming module
        from streaming import StreamingOutput
        
        # Create a streaming output handler if it doesn't exist
        if not hasattr(self, 'streaming_output'):
            self.streaming_output = StreamingOutput()
        
        # Start timing for response time tracking
        start_time = time.time()
        
        # Detect language if not specified
        if language is None:
            language = self._detect_language(query)
        
        # Store the message in conversation history if user_id is provided
        if user_id:
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []
            
            # Add the user message to the conversation history
            self.conversation_history[user_id].append({
                "role": "user",
                "content": query,
                "timestamp": time.time(),
                "platform": platform
            })
        
        # Find relevant information from the knowledge base
        relevant_info = self.find_relevant_information(query)
        
        # Determine the topic of the query
        topic = self._determine_topic(query)
        
        # Extract contexts from relevant information
        contexts = [info["text"] for info in relevant_info]
        context_text = "\n\n".join(contexts)
        
        # Create a system prompt with the knowledge context
        system_prompt = f"""Ты - официальный AI-ассистент проекта Lucky Train на блокчейне TON. 
        Твоя задача - предоставлять точную и полезную информацию о проекте, его токене, метавселенной и блокчейне TON.
        
        Используй следующую информацию для ответа:
        
        {context_text}
        
        Тема запроса: {topic}
        
        Говори уверенно и профессионально. Если информации недостаточно, вежливо скажи, что у тебя нет полных данных по этому вопросу.
        Не выдумывай информацию. Отвечай на языке: {self.supported_languages.get(language, "русском")} используя соответствующие языковые и культурные особенности."""
        
        # Create message structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Add conversation context if available
        if user_id:
            conversation_context = self.get_conversation_context(user_id, max_messages=3)
            if conversation_context:
                # Replace the messages with system prompt + conversation context + current query
                messages = [{"role": "system", "content": system_prompt}] + conversation_context
        
        # Get model configuration
        model = self.config.get("llm_settings", {}).get("llm_model", "gpt-3.5-turbo")
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 500)
        
        try:
            # Handle different platforms
            if platform == "telegram" and bot and chat_id:
                # Stream to Telegram
                full_response = self.streaming_output.stream_to_telegram(
                    openai_client=self.openai_client,
                    bot=bot,
                    chat_id=chat_id,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Calculate response time and track interaction
                response_time = time.time() - start_time
                self._track_interaction(user_id, platform, query, full_response or "", response_time)
                
                return None  # Streaming is handled asynchronously
                
            elif platform == "website" and websocket_session_id:
                # Stream to WebSocket
                self.streaming_output.stream_to_websocket(
                    openai_client=self.openai_client,
                    session_id=websocket_session_id,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Tracking is handled in the streaming callback
                return None  # Streaming is handled asynchronously
                
            elif platform == "console":
                # Stream to console
                self.streaming_output.stream_to_console(
                    openai_client=self.openai_client,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return None  # Streaming is handled asynchronously
                
            else:
                # Fall back to non-streaming
                logger.warning("Streaming not available for this platform or missing parameters, falling back to non-streaming")
                return self.handle_message(query, user_id, platform)
                
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            
            # Track the error
            self.analytics_manager.track_error(
                error_type="streaming",
                error_message=str(e),
                user_id=user_id,
                platform=platform
            )
            
            # Fall back to non-streaming
            return self.handle_message(query, user_id, platform)

# Example usage
if __name__ == "__main__":
    assistant = LuckyTrainAssistant()
    
    # Example queries
    queries = [
        "Что такое Lucky Train?",
        "Расскажи о TON блокчейне",
        "Как используется токен Lucky Train?",
        "Что такое метавселенная Lucky Train?"
    ]
    
    for query in queries:
        print(f"Запрос: {query}")
        response = assistant.handle_message(query)
        print(f"Ответ: {response}")
        print() 