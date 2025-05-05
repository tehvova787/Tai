"""
Telegram Bot for Lucky Train AI Assistant

This module provides a Telegram bot interface for the Lucky Train AI assistant.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
import json
import time
import sys

# Add the parent directory to the Python path to import the assistant module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bot.assistant import LuckyTrainAssistant
from analytics import FeedbackCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LuckyTrainTelegramBot:
    """Telegram bot for the Lucky Train AI assistant."""
    
    def __init__(self, token: str, config_path: str = "../config/config.json"):
        """Initialize the Telegram bot.
        
        Args:
            token: The Telegram bot token.
            config_path: Path to the configuration file.
        """
        self.token = token
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize the assistant
        self.assistant = LuckyTrainAssistant(config_path)
        
        # Get Telegram settings
        self.telegram_settings = self.config.get("telegram_settings", {})
        self.welcome_message = self.telegram_settings.get("welcome_message", 
            "Привет! Я официальный AI-ассистент проекта Lucky Train на блокчейне TON. Я могу рассказать вам о проекте, токене LTT, метавселенной и многом другом.")
        
        # Get commands
        self.commands = self.telegram_settings.get("commands", {
            "start": "Начать общение с ботом",
            "help": "Показать список доступных команд",
            "about": "Информация о проекте Lucky Train",
            "token": "Информация о токене LTT",
            "metaverse": "Информация о метавселенной Lucky Train"
        })
        
        # Initialize rate limiting
        self.user_last_message_time = {}
        self.rate_limit_seconds = self.config.get("security_settings", {}).get("rate_limit", {}).get("max_requests_per_minute", 60)
        if self.rate_limit_seconds > 0:
            self.rate_limit_seconds = 60 / self.rate_limit_seconds  # Convert to seconds between requests
        
        # Check if streaming is enabled
        self.streaming_enabled = self.config.get("llm_settings", {}).get("streaming", False)
        
        # Initialize feedback collector
        self.feedback_collector = FeedbackCollector(self.assistant.analytics_manager)
        
        logger.info("Telegram bot initialized successfully")
    
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
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command.
        
        Args:
            update: The Telegram update.
            context: The callback context.
        """
        user = update.effective_user
        logger.info(f"User {user.id} started the bot")
        
        # Create inline keyboard buttons
        keyboard = [
            [
                InlineKeyboardButton("О проекте", callback_data="about"),
                InlineKeyboardButton("Токен LTT", callback_data="token")
            ],
            [
                InlineKeyboardButton("Метавселенная", callback_data="metaverse"),
                InlineKeyboardButton("Помощь", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send welcome message with buttons
        await update.message.reply_text(
            self.welcome_message,
            reply_markup=reply_markup
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /help command.
        
        Args:
            update: The Telegram update.
            context: The callback context.
        """
        user = update.effective_user
        logger.info(f"User {user.id} requested help")
        
        # Create a help message with available commands
        help_message = "🤖 Доступные команды:\n\n"
        
        for command, description in self.commands.items():
            help_message += f"/{command} - {description}\n"
        
        help_message += "\nВы также можете просто задать мне вопрос о проекте Lucky Train, токене LTT, метавселенной или блокчейне TON."
        
        await update.message.reply_text(help_message)
    
    async def about_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /about command.
        
        Args:
            update: The Telegram update.
            context: The callback context.
        """
        user = update.effective_user
        logger.info(f"User {user.id} requested information about the project")
        
        response = self.assistant.handle_message("Расскажи о проекте Lucky Train", str(user.id), "telegram")
        
        await update.message.reply_text(response)
    
    async def token_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /token command.
        
        Args:
            update: The Telegram update.
            context: The callback context.
        """
        user = update.effective_user
        logger.info(f"User {user.id} requested information about the token")
        
        response = self.assistant.handle_message("Расскажи о токене LTT", str(user.id), "telegram")
        
        await update.message.reply_text(response)
    
    async def metaverse_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /metaverse command.
        
        Args:
            update: The Telegram update.
            context: The callback context.
        """
        user = update.effective_user
        logger.info(f"User {user.id} requested information about the metaverse")
        
        response = self.assistant.handle_message("Расскажи о метавселенной Lucky Train", str(user.id), "telegram")
        
        await update.message.reply_text(response)
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle callback queries from inline keyboard buttons.
        
        Args:
            update: The Telegram update.
            context: The callback context.
        """
        query = update.callback_query
        await query.answer()
        
        user = query.from_user
        logger.info(f"User {user.id} clicked button: {query.data}")
        
        # Check if this is a feedback button
        if query.data.startswith("feedback_"):
            # Extract rating from callback data
            parts = query.data.split("_")
            if len(parts) >= 3:
                message_id = parts[1]
                rating = int(parts[2]) if parts[2].isdigit() else 3
                
                # Process the feedback
                self.feedback_collector.process_feedback(
                    user_id=str(user.id),
                    platform="telegram",
                    message_id=message_id,
                    rating=rating
                )
                
                # Send acknowledgment
                await query.edit_message_text(
                    text="Спасибо за ваш отзыв! Это помогает нам улучшать качество ответов."
                )
                return
        
        if query.data == "about":
            response = self.assistant.handle_message("Расскажи о проекте Lucky Train", str(user.id), "telegram")
            await query.edit_message_text(text=response)
            
        elif query.data == "token":
            response = self.assistant.handle_message("Расскажи о токене LTT", str(user.id), "telegram")
            await query.edit_message_text(text=response)
            
        elif query.data == "metaverse":
            response = self.assistant.handle_message("Расскажи о метавселенной Lucky Train", str(user.id), "telegram")
            await query.edit_message_text(text=response)
            
        elif query.data == "help":
            # Create a help message with available commands
            help_message = "🤖 Доступные команды:\n\n"
            
            for command, description in self.commands.items():
                help_message += f"/{command} - {description}\n"
            
            help_message += "\nВы также можете просто задать мне вопрос о проекте Lucky Train, токене LTT, метавселенной или блокчейне TON."
            
            await query.edit_message_text(text=help_message)
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if the user has exceeded the rate limit.
        
        Args:
            user_id: The user's ID.
            
        Returns:
            True if the user is within the rate limit, False otherwise.
        """
        # If rate limiting is disabled, always allow
        if self.rate_limit_seconds <= 0:
            return True
        
        current_time = time.time()
        last_message_time = self.user_last_message_time.get(user_id, 0)
        
        # Check if enough time has passed since the last message
        if current_time - last_message_time < self.rate_limit_seconds:
            return False
        
        # Update the last message time
        self.user_last_message_time[user_id] = current_time
        return True
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages.
        
        Args:
            update: The Telegram update.
            context: The callback context.
        """
        user = update.effective_user
        user_id = str(user.id)
        message_text = update.message.text
        
        logger.info(f"Received message from user {user_id}: {message_text}")
        
        # Check rate limiting
        if not self._check_rate_limit(user_id):
            logger.warning(f"User {user_id} exceeded rate limit")
            await update.message.reply_text(
                "Пожалуйста, не отправляйте сообщения слишком часто. Подождите немного и попробуйте снова."
            )
            return
        
        # Check if streaming is enabled and the assistant supports it
        if self.streaming_enabled and hasattr(self.assistant, 'stream_response'):
            try:
                # Use streaming response
                # This will handle the response asynchronously
                self.assistant.stream_response(
                    query=message_text,
                    user_id=user_id,
                    platform="telegram",
                    bot=context.bot,
                    chat_id=update.effective_chat.id
                )
            except Exception as e:
                logger.error(f"Error streaming response: {e}")
                # Fall back to non-streaming response with feedback buttons
                try:
                    response = self.assistant.handle_message(message_text, user_id, "telegram")
                    
                    # Generate a unique message ID for feedback
                    message_id = f"tg_{user_id}_{int(time.time())}"
                    
                    # Create feedback buttons
                    keyboard = [
                        [
                            InlineKeyboardButton("👍", callback_data=f"feedback_{message_id}_5"),
                            InlineKeyboardButton("👎", callback_data=f"feedback_{message_id}_1")
                        ]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await update.message.reply_text(response, reply_markup=reply_markup)
                except Exception as e:
                    logger.error(f"Error handling fallback response: {e}")
                    await update.message.reply_text("Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз.")
        else:
            # Use standard response with feedback buttons
            try:
                # Get the response
                response = self.assistant.handle_message(message_text, user_id, "telegram")
                
                # Generate a unique message ID for feedback
                message_id = f"tg_{user_id}_{int(time.time())}"
                
                # Create feedback buttons
                keyboard = [
                    [
                        InlineKeyboardButton("👍", callback_data=f"feedback_{message_id}_5"),
                        InlineKeyboardButton("👎", callback_data=f"feedback_{message_id}_1")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(response, reply_markup=reply_markup)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                await update.message.reply_text("Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз.")
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors.
        
        Args:
            update: The Telegram update.
            context: The callback context.
        """
        logger.error(f"Error: {context.error} - {type(context.error)}")
        
        if update and isinstance(update, Update) and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз позже."
            )
    
    def run(self) -> None:
        """Run the Telegram bot."""
        # Create the application
        application = ApplicationBuilder().token(self.token).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("about", self.about_command))
        application.add_handler(CommandHandler("token", self.token_command))
        application.add_handler(CommandHandler("metaverse", self.metaverse_command))
        
        # Add callback query handler for inline keyboard buttons
        application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Add message handler
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Add error handler
        application.add_error_handler(self.error_handler)
        
        # Start the bot
        logger.info("Starting Telegram bot...")
        application.run_polling()

# Example usage
if __name__ == "__main__":
    # Get the Telegram bot token from environment variable
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set")
        print("Please set the TELEGRAM_BOT_TOKEN environment variable and try again.")
        sys.exit(1)
    
    bot = LuckyTrainTelegramBot(token)
    bot.run() 